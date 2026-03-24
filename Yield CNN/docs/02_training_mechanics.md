# Training mechanics

Author: Reynaldo Gomez
Semiconductor-Engineering: Yield CNN

---

## The training loop

Every epoch, for every batch drawn from the training loader, the model executes five
operations in fixed order:

```python
optimizer.zero_grad(set_to_none=True)   # clear accumulated gradients
logits = model(x)                        # forward pass: x -> 9-class scores
loss   = loss_fn(logits, y)              # scalar loss over the batch
loss.backward()                          # backprop: compute dL/dtheta for all theta
optimizer.step()                         # update: move each weight by -lr * gradient
```

These five lines are the complete optimization engine. The remainder of the training
script, covering data loading, checkpoint saving, metric logging, and scheduler updates, is
scaffolding that supports them.

---

## Loss functions

### Cross-entropy loss

For a single sample, CrossEntropyLoss computes:

$$
L_{\text{CE}} = -\log(p_t)
$$

where `p_t` is the softmax probability the model assigns to the correct class. A
confident correct prediction (`p_t = 0.95`) contributes `-log(0.95) = 0.05` to the
loss; a near-random wrong prediction (`p_t = 0.05`) contributes `-log(0.05) = 3.0`.
The gradient of the loss with respect to the logits is large when the model is wrong
and small when it is right, which is the intended behavior.

The problem on LSWMD is one of scale. The model reaches high confidence on *none*
(29,486 validation samples) early in training, reducing the per-sample CE contribution
for that class to near zero. But 29,486 near-zero gradients still numerically dominate
the batch gradient over the 111 Donut or 239 Scratch samples that each contribute full
loss. The optimizer allocates the majority of its gradient budget to refining *none*
predictions that are already nearly correct.

### Focal loss

FocalLoss modifies CrossEntropyLoss by multiplying the per-sample loss by a
confidence-dependent weighting term:

$$
L_{\text{FL}} = -(1 - p_t)^{\gamma} \log(p_t)
$$

The factor `(1 - p_t)^gamma` is the focal weight. With gamma=2.0: a sample correctly
predicted with probability 0.95 receives weight `(0.05)^2 = 0.0025`, suppressing its
contribution to near zero; a sample predicted at chance (`p_t = 0.50`) receives weight
`(0.50)^2 = 0.25`; a hard misclassification (`p_t = 0.05`) receives weight
`(0.95)^2 = 0.9025`, preserving nearly the full CE gradient. The 29,486 easy *none*
samples that would otherwise dominate gradient updates are reduced to negligible
contributors, and the optimizer's effective budget is concentrated on Donut, Scratch,
and the other spatially ambiguous defect types. This mechanism accounts for the Donut
F1 improvement from 0.56 to 0.89 between the baseline and ResNet+Focal experiments.

The gamma=2.0 value is empirically established in the original paper. Reducing gamma
smoothly recovers standard CrossEntropyLoss at gamma=0. Increasing gamma
over-suppresses easy samples; in practice, values above 5 tend to destabilize training
as the gradient signal becomes dominated by a small number of hard examples.

Original paper: https://arxiv.org/abs/1708.02002

---

## Backpropagation

After the forward pass produces a scalar loss `L`, PyTorch's autograd engine computes
the gradient of `L` with respect to every trainable parameter via the chain rule.
For a weight matrix `W_l` in layer `l` of an `L`-layer network:

$$
\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial z_L} \cdot \frac{\partial z_L}{\partial z_{L-1}} \cdots \frac{\partial z_{l+1}}{\partial z_l} \cdot \frac{\partial z_l}{\partial W_l}
$$

Each `dz_(k+1)/dz_k` is the Jacobian of one layer's output with respect to its input,
evaluated at the current forward-pass activations. PyTorch records the computation
graph during the forward pass and traverses it in reverse during `loss.backward()` to
accumulate these products.

The vanishing gradient problem is the numerical consequence of this product structure:
if each intermediate Jacobian has singular values slightly below 1, the product of `L`
Jacobians decays exponentially with depth, and early-layer weights receive gradients
too small to produce meaningful updates. Skip connections address this by adding a +1
term to the gradient at each residual block (see `01_cnn_fundamentals.md`), guaranteeing
a lower bound on gradient magnitude that does not depend on the learned function `F(x)`.

---

## Optimizer: AdamW

After `loss.backward()` populates `param.grad` for every parameter, `optimizer.step()`
applies the AdamW update rule:

$$
\theta_{t+1} = \theta_t - \text{lr} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \text{lr} \cdot \lambda \cdot \theta_t
$$

`m_hat_t` is the bias-corrected first moment (running mean of gradients across past
batches), `v_hat_t` is the bias-corrected second moment (running mean of squared
gradients), and the final term `- lr * lambda * theta_t` is the weight decay correction.
In the training scripts, `lr = 3e-4`, `epsilon = 1e-8`, and `lambda = 1e-4`.

The first moment smooths the update direction across noisy batches; the second moment
adaptively reduces the learning rate for parameters whose gradients have high variance
across batches, stabilizing training when different parts of the network, such as the
feature extractor and the classifier head, are learning at different rates. The weight
decay term pulls every weight slightly toward zero each step, providing L2 regularization.
Standard Adam applies weight decay incorrectly by folding it into the adaptive scaling
term; AdamW decouples them, which Loshchilov & Hutter show produces better regularization.

AdamW paper: https://arxiv.org/abs/1711.05101

---

## Learning rate schedulers

The learning rate controls the magnitude of each weight update step. A rate that is too
large causes the optimizer to overshoot loss minima and diverge; one that is too small
produces progress too slow to be practical. Schedulers decay the learning rate over
training so that early steps make large, fast progress and late steps fine-tune the
solution near convergence.

### ReduceLROnPlateau

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)
scheduler.step(va_loss)   # called once per epoch
```

This scheduler monitors `va_loss` after each epoch. If the validation loss fails to
improve for three consecutive epochs, the learning rate is multiplied by 0.5. The
behavior is reactive, responding to what the training dynamics are actually doing,
which makes it robust to datasets with irregular convergence curves. The limitation
is that the halving event is abrupt; a sudden factor-of-two reduction can perturb the
optimizer's accumulated momentum estimates (`m_hat`, `v_hat`) and cause a transient
loss spike before the reduced rate stabilizes.

### CosineAnnealingLR

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)
scheduler.step()   # no argument; steps on epoch index
```

The learning rate follows a cosine curve from its initial value to `eta_min` over
`T_max` epochs:

$$
\text{lr}_t = \eta_{\min} + \frac{1}{2}(\text{lr}_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi t}{T_{\max}}\right)\right)
$$

The decay is smooth with no abrupt reductions, which tends to be more stable than
ReduceLROnPlateau for longer training runs where momentum estimates carry useful history.

### CosineAnnealingWarmRestarts

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

SGDR applies cosine decay for `T_0` epochs and then resets the learning rate to
`lr_max`, repeating with each cycle length multiplied by `T_mult`. For `T_0=10,
T_mult=2`, the restart schedule is 10 epochs, 20 epochs, 40 epochs. The restart jolts
the optimizer out of the local minimum it has settled into; the subsequent cosine decay
then fine-tunes within the new region of the loss landscape. Checkpoints should be
saved at the end of each cosine cycle, when the learning rate is near `eta_min` and the
model is most thoroughly exploiting the current basin.

Original paper: https://arxiv.org/abs/1608.03983

---

## WeightedRandomSampler

The LSWMD class distribution spans nearly three orders of magnitude: *none* accounts
for 147,431 training samples while *Near-full* provides 149. Without intervention, the
training loader draws batches that reflect this distribution, and the model converges
toward a solution that is highly accurate on *none* but has seen too few rare-class
examples to form reliable representations for them.

`WeightedRandomSampler` assigns each training sample a draw probability inversely
proportional to its class frequency:

```python
class_weights  = 1.0 / class_counts
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
```

*Near-full* receives weight `1/149 = 0.0067`; *none* receives `1/147,431 = 0.0000068`.
Each *Near-full* sample is drawn approximately 990× more often per unit probability
than each *none* sample. The effect is that every mini-batch contains roughly equal
representation across all nine classes regardless of the underlying dataset distribution.

This mechanism operates at the data pipeline level, changing which samples appear in
each batch. Loss class weighting, by contrast, operates at the gradient level, changing
how much each sample's loss contributes to the weight update even after it has
been sampled. The two can be combined: WeightedRandomSampler is already active in the
current scripts; loss class weights are an additional term that can be passed to
`FocalLoss` if per-class gradient amplification is needed beyond what the sampler provides.

---

## Dropout

```python
self.dropout = nn.Dropout(0.4)
```

During the training forward pass, Dropout randomly zeroes 40% of the 256 post-pool
features before they reach the linear classifier. The zeroed features vary randomly
per forward pass, forcing the network to distribute the discriminative information for
each class across multiple redundant feature channels rather than concentrating it in
a small number of high-confidence neurons. During evaluation (`model.eval()`), Dropout
is disabled and all 256 features pass through unchanged, recovering the full
representational capacity of the learned feature set.

---

## What overfitting looks like

Training diagnostics are visible in the epoch-by-epoch output:

```
Epoch 01 | train loss 0.45 acc 0.85 | val loss 0.38 acc 0.88   <- underfitting
Epoch 10 | train loss 0.12 acc 0.96 | val loss 0.22 acc 0.96   <- healthy
Epoch 18 | train loss 0.01 acc 0.99 | val loss 0.25 acc 0.97   <- slight overfit
Epoch 40 | train loss 0.00 acc 1.00 | val loss 0.45 acc 0.94   <- overfit
```

The canonical overfit signature is a training loss that continues to fall while validation
loss stabilizes or rises, accompanied by a widening gap between training and validation
accuracy. The model is memorizing training-set patterns that do not generalize. In the
observed runs at epoch 20: train loss 0.0093, val loss 0.2774. The gap is large, but
validation accuracy remains high (0.97) and the model generalizes well in practice.
This is the common regime with dropout and BatchNorm active: train loss reaches near-zero
because the training set has been seen many times, but the regularization prevents the
learned representations from becoming training-set-specific.