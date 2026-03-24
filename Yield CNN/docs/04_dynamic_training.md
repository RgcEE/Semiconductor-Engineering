# Dynamic training techniques

Author: Reynaldo Gomez
Semiconductor-Engineering: Yield CNN

This document covers adaptive training, specifically the "differential score analyzer" concept, and connects it to real implemented techniques. It also addresses the training black box problem and the instrumentation needed to open it.

---

## The idea, formally stated

The goal is to monitor per-class training signals, detect when individual classes are
not converging, intervene by adjusting loss weights or optimizer state, save model
state before the intervention, and revert if the intervention makes things worse over
the subsequent evaluation window. This maps to three concrete implementations: dynamic
class weighting via convergence-rate monitoring, learning rate warm restarts (SGDR) to
escape local minima when global progress stalls, and checkpoint-and-branch to make
interventions reversible.

---

## 1. Dynamic class weighting

Standard CrossEntropyLoss treats all classes with equal per-sample weight. FocalLoss
down-weights easy samples globally through the confidence term `(1 - p_t)^gamma`.
Dynamic class weighting extends this by adjusting the per-class loss contribution in
response to each class's observed convergence rate during training.

Define the per-class F1 improvement rate over a window of W epochs:

$$
\Delta F_{1,c}(t) = F_{1,c}(t) - F_{1,c}(t - W)
$$

When `delta_F1_c(t) < epsilon`, meaning the improvement falls below a convergence threshold,
class `c` is considered stalled and receives a multiplicative weight boost:

$$
w_c(t+1) = w_c(t) \cdot \left(1 + \beta \cdot \mathbf{1}\!\left[\Delta F_{1,c}(t) < \epsilon\right]\right)
$$

`beta` controls the aggressiveness of the boost (0.1 to 0.5 are reasonable starting
values); `epsilon` defines the convergence threshold (0.005 represents less than 0.5
percentage-point F1 improvement over W=3 epochs). These weights are passed directly
to the loss function as class-level scaling factors:

```python
class_weights = torch.tensor(w_c, dtype=torch.float32).to(DEVICE)
loss = F.cross_entropy(logits, targets, weight=class_weights)
```

An implementation sketch for the main training loop:

```python
class_boost = {cls: 1.0 for cls in classes}
f1_history  = {cls: [] for cls in classes}

for epoch in range(1, EPOCHS + 1):
    train_one_epoch(...)
    va_loss, va_acc = eval_one_epoch(...)

    y_pred, y_true = collect_predictions(model, val_loader)
    report = classification_report(y_true, y_pred,
                                   target_names=classes,
                                   output_dict=True)

    for cls in classes:
        f1_history[cls].append(report[cls]["f1-score"])

    if epoch >= 4:
        for cls in classes:
            delta = f1_history[cls][-1] - f1_history[cls][-4]
            if delta < 0.005:
                class_boost[cls] = min(class_boost[cls] * 1.2, 10.0)
                print(f"  [boost] {cls}: weight -> {class_boost[cls]:.2f}")

    weights = torch.tensor([class_boost[c] for c in classes],
                            dtype=torch.float32).to(DEVICE)
    loss_fn = FocalLoss(gamma=2.0, class_weights=weights)
```

The cap at 10.0 prevents runaway amplification. The risk is that a class with low F1
due to genuine ambiguity, such as insufficient training data or overlapping feature distributions,
will receive escalating weight that destabilizes training rather than resolves the
underlying problem. Monitoring validation loss alongside the boosted weights is essential;
if validation loss begins rising while the boost is active, the intervention is causing
harm rather than targeted improvement.

---

## 2. Learning rate warm restarts (SGDR)

The cosine annealing schedule decays the learning rate from `eta_max` to `eta_min` over
`T_max` epochs:

$$
\text{lr}(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi t}{T_{\max}}\right)\right)
$$

SGDR resets `t` to zero at the end of each cycle, jumping the learning rate back to
`eta_max`, and multiplies the cycle length by `T_mult` so that successive cycles grow
longer. The argument for restarts is geometric: the loss landscape is not a smooth
bowl but a high-dimensional surface with many local minima separated by loss barriers.
Late in a cosine cycle with a small learning rate, the optimizer is settled in whatever
basin it reached; it has no mechanism to escape. A warm restart with `lr = eta_max`
provides the kinetic energy to cross low barriers and sample a different region of the
landscape, after which cosine decay settles the model into the best minimum found in
that region. The final saved checkpoint should come from the end of the last cosine
cycle, when the learning rate is near `eta_min` and the model is most fully settled.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0    = 10,    # first restart after 10 epochs
    T_mult = 2,     # each subsequent cycle: 10, 20, 40 epochs
    eta_min = 1e-6
)

# per-epoch step:
scheduler.step()
```

Original paper: https://arxiv.org/abs/1608.03983

---

## 3. Checkpoint-and-branch

Before applying any intervention, whether a class weight boost, a learning rate change, or a
hyperparameter modification, save a complete snapshot of the training state. Apply
the intervention and train for N evaluation epochs. If macro F1 after the intervention
is below macro F1 before it by more than the measurement noise threshold, restore the
saved state and try a different intervention.

The critical implementation detail is saving the optimizer state alongside the model
weights. Adam maintains per-parameter momentum estimates `m_hat` and `v_hat` that
encode the gradient history accumulated over many batches. Loading only model weights
while initializing a fresh optimizer discards this history and causes a transient loss
spike as the optimizer re-estimates the gradient statistics from scratch. Saving and
restoring `optimizer.state_dict()` preserves the accumulated history:

```python
import copy

pre_intervention = {
    "model":     copy.deepcopy(model.state_dict()),
    "optimizer": copy.deepcopy(optimizer.state_dict()),
    "scheduler": copy.deepcopy(scheduler.state_dict()),
    "epoch":     epoch,
    "macro_f1":  macro_f1_before,
}
torch.save(pre_intervention, "checkpoints/pre_intervention.pt")

# apply intervention, train N epochs, evaluate
# if macro_f1_after < macro_f1_before - 0.01:
#     model.load_state_dict(pre_intervention["model"])
#     optimizer.load_state_dict(pre_intervention["optimizer"])
# print("Reverted: intervention made things worse")
```

---

## 4. Opening the black box

The default epoch-level output reports only aggregate train loss, train accuracy,
validation loss, and validation accuracy. This is not sufficient to diagnose why a
class is failing to converge or why an intervention is not working.

**Per-batch loss logging** reveals whether the loss is descending smoothly within an
epoch or oscillating, a symptom of a learning rate that is too large or of gradient
estimates that are too noisy. Adding a print every 50 batches within the training loop
is sufficient for diagnostics:

```python
if batch_idx % 50 == 0:
    print(f"  Batch {batch_idx}/{len(loader)} loss {loss.item():.4f}")
```

**Gradient norm monitoring** reveals whether the optimizer is receiving useful signal.
After `loss.backward()` but before `optimizer.step()`, compute the global gradient norm
across all parameters:

```python
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
```

A gradient norm substantially above 10 indicates the update is unstable; below 0.001
indicates the signal has vanished. Gradient clipping is the standard fix for the former:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# insert between loss.backward() and optimizer.step()
```

**Activation statistics** detect saturation or collapse in intermediate representations.
A forward hook on any layer reports its output distribution without modifying the forward
pass:

```python
def hook_fn(module, input, output):
    print(f"  Layer output: mean={output.mean():.4f} std={output.std():.4f}")

handle = model.features[0].register_forward_hook(hook_fn)
# run one batch; then handle.remove()
```

Activations with near-zero mean and near-zero standard deviation indicate saturation
or dead neurons. Activations with mean or standard deviation significantly above 1
indicate BatchNorm is not functioning correctly or the learning rate is too high.

**Grad-CAM** provides spatial attribution of model decisions at the level of individual
predictions. The gradient of the class score with respect to the last convolutional
layer's feature maps identifies which spatial regions most influenced the prediction.
If a model predicts Edge-Ring correctly but Grad-CAM highlights the center of the wafer,
the model has learned a spurious correlation rather than the ring structure, which is the correct
feature for that class. This is the most direct available check that the model is not
solving the classification problem through an unintended shortcut.

Paper: https://arxiv.org/abs/1610.02391

---

## 5. Connecting this to research

The "differential score analyzer" concept most closely corresponds to three published
methods.

**Class-Balanced Loss** (Cui et al., 2019) reweights the per-sample loss by the
effective number of samples for each class, defined as `(1 - beta^n) / (1 - beta)` where
`n` is the class count and `beta` approaches 1. The effective number saturates for large
`n` and provides a more principled weighting than `1/n`, which over-amplifies the rarest
classes at high imbalance ratios. https://arxiv.org/abs/1901.05555

**Online Hard Example Mining (OHEM)** (Shrivastava et al., 2016) selects the K
highest-loss samples from each batch for backpropagation, discarding the easy samples
entirely. The selection is dynamic and per-batch; no per-class bookkeeping is required.
This concentrates gradient signal on hard examples automatically without needing to
track convergence rates explicitly. https://arxiv.org/abs/1604.03540

**Meta-Weight-Net** (Shu et al., 2019) trains a second small network that takes a
sample's gradient as input and outputs a scalar loss weight. The weight network is
trained on a small clean meta-validation set held out from the main training data,
using the meta-gradient to optimize the loss weighting for generalization rather than
training fit. This is the closest published method to the adaptive score-based weighting
described above. https://arxiv.org/abs/1902.07379

**Population Based Training** (Jaderberg et al., 2017, DeepMind) addresses the
checkpoint-and-branch concept at scale: a population of models is trained simultaneously,
and weights from higher-performing models are periodically copied to lower-performing
ones with hyperparameter mutations applied. It automates the branching and comparison
process across a model population rather than applying it to a single training run.
https://arxiv.org/abs/1711.09846

The fundamental constraint on building adaptive training for a single run is that any
intervention cannot be evaluated until several epochs have elapsed, by which point the
training trajectory has already diverged from the pre-intervention baseline. The
practical resolution is either to run multiple parallel configurations and compare
outcomes, or to maintain a held-out meta-validation set used exclusively for evaluating
interventions while keeping the main validation set clean for final experimental
comparison.

---

## Practical next step

Before building the adaptive system, the training loop needs to produce the data
required to make intervention decisions. Instrumenting `yield_resnet_focal.py` with
per-epoch per-class F1 logging, computed every five epochs to avoid the overhead of
full inference on every epoch, provides the convergence signal that the adaptive logic
would consume:

```python
epoch_log = []

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(...)
    va_loss, va_acc = eval_one_epoch(...)

    if epoch % 5 == 0:
        y_pred, y_true = collect_predictions(model, val_loader)
        report = classification_report(y_true, y_pred,
                                       target_names=classes,
                                       output_dict=True)
        per_class_f1 = {c: report[c]["f1-score"] for c in classes}
        epoch_log.append({"epoch": epoch, "va_loss": va_loss, **per_class_f1})
        print(f"  Per-class F1: {per_class_f1}")

import json
with open("epoch_log.json", "w") as f:
    json.dump(epoch_log, f, indent=2)
```

Once this log exists across several runs, the epochs at which each class stalls, the
epochs that are most productive, and the points at which interventions would have the
highest expected impact all become observable from data. Adaptive logic built from
those observed patterns will be better calibrated than logic built from theoretical
estimates alone.