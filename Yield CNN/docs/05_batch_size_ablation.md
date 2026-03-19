# Batch Size and Learning Rate Ablation: yield_resnet_focal
**Author: Reynaldo Gomez**
**Date: 3/18/2026**
**Repo: Semiconductor-Engineering / Yield CNN / docs**

---

## What Batch Size Controls

Batch size determines how many samples the model sees before each weight update.

With 138,360 training samples:

```
batch=64:  138,360 / 64  = 2,162 weight updates per epoch
batch=128: 138,360 / 128 = 1,081 weight updates per epoch
batch=256: 138,360 / 256 =   540 weight updates per epoch
```

The same data is seen every epoch regardless of batch size. What changes is
how frequently the model updates its weights and how noisy each gradient
estimate is.

Each gradient update is computed as the average loss over one batch. A batch
of 64 samples gives a noisier estimate of the true gradient than a batch of
128. That noise has two effects that pull in opposite directions:

**Noise hurts:** Noisier gradient estimates mean each step is less accurate.
The model may move in a slightly wrong direction, slowing convergence.

**Noise helps:** Random perturbations in the gradient help the optimizer
escape shallow local minima and find flatter, more generalizable solutions.
This is called the implicit regularization effect of small batch training.
A model trained with batch=64 often generalizes better than one trained with
batch=256 because the noise prevented it from settling into sharp minima that
overfit to the training set.

For rare classes specifically (Scratch: 1,193 total samples; Donut: 555;
Near-full: 149), the optimizer needed this noisy exploration to find good
representations. Removing it with batch=256 collapsed Scratch precision to
near-random levels across both runs.

---

## What Learning Rate Controls

The learning rate scales each weight update step:

```
theta = theta - LR * gradient
```

Think of the loss landscape as a hillside with valleys. LR controls how far
you step downhill each update:

```
LR too large:  overshoot the valley, bounce around, never settle
LR too small:  take tiny steps, converge slowly, get stuck in shallow dips
LR just right: descend efficiently and settle in the best valley
```

---

## The Linear Scaling Rule

When you change batch size, learning rate should scale in the same direction
to keep the effective learning signal per epoch roughly constant.

```
new_LR = old_LR * (new_batch / old_batch)
```

Halving batch size means halving LR:

```
batch=128, LR=3e-4  ->  batch=64, LR=1.5e-4
```

1.5e-4 = 0.00015 is smaller than 3e-4 = 0.0003. Halving batch size means
halving LR, not increasing it.

Why: with smaller batches each gradient estimate is noisier. A smaller LR
compensates by taking more conservative steps on that noisier estimate.
The total effective learning per epoch stays approximately the same:

```
Effective update per epoch ~= LR * gradient * updates_per_epoch

batch=128, LR=3e-4:   3e-4   * gradient * 1,081 steps
batch=64,  LR=1.5e-4: 1.5e-4 * gradient * 2,162 steps  <- same order
```

This rule is a heuristic, not a law. It is a principled starting point.

Reference: Goyal et al. 2017, Accurate, Large Minibatch SGD.
https://arxiv.org/abs/1706.02677

---

## All Runs: Full Results

```
batch   epochs   LR      best_ep   val_loss   macro_f1   scratch_f1
-----   ------   ------  -------   --------   --------   ----------
128       20     3e-4      18       0.2540      0.869      0.705
128       40     3e-4      12       0.2870      0.888      0.803   <- best
256       20     3e-4      18       0.2544      0.863      0.689
256       20     3e-4      12       0.2520      0.851      0.576
 64       20     3e-4      20       0.2751      0.863      0.691
 64       20     1.5e-4    20       0.3382      0.860      0.678
 64       40     1.5e-4     8       0.3210      0.871      0.773
```

---

## What the Results Show

**256 batch consistently degraded performance.**
Scratch precision collapsed to 0.44-0.60 across both runs. Two confirming
runs is sufficient evidence; the pattern is real, not random seed variance.
Fewer gradient updates (540 per epoch) combined with reduced batch noise
removed the implicit regularization the rare classes depended on.

**64 batch did not improve on 128.**
Macro F1 across all 64 runs landed at 0.860-0.871, below the 128/40 best
of 0.888. The extra gradient noise from smaller batches did not help here.
With WeightedRandomSampler active and 9 classes, batch=64 gives roughly
7 samples per class per batch. For Near-full (149 total samples) this is
an extremely noisy estimate for BatchNorm statistics, which partially offset
any regularization benefit from the smaller batch.

**64/40 epochs at LR=1.5e-4 peaked at epoch 8.**
Lower LR caused slower initial descent. The model converged at a shallower
point earlier and then overfitted for 32 epochs. Best epoch 8 out of 40
means most of the run was wasted compute. When train loss keeps falling while
val loss stays flat or rises, the model has memorized the training set;
additional epochs are not helping generalization.

**Val loss and macro F1 are decoupled.**
The lowest val loss (0.2520) came from the worst macro F1 run (0.851).
Val loss is dominated by the none class (85% of validation samples). A model
that trades a small drop in none performance for better rare class performance
will show higher val loss but better macro F1. Save checkpoints based on val
loss (stable, differentiable), but compare experiments based on macro F1.

**On experimental controls.**
A single training run does not tell you if a configuration is genuinely better
or just got lucky with random initialization and batch ordering. The rigorous
approach is three runs per configuration, comparing the distribution of results.
The practical approach: if two runs of a configuration both show degradation on
the metrics you care about, that is sufficient evidence to move on. If results
are mixed after two runs, run a third. Do not run infinite combinations hoping
for a lucky result; that is not engineering, that is guessing.

---

## Confirmed Optimal Configuration

```
batch_size = 128
LR         = 3e-4
epochs     = 40
optimizer  = AdamW, weight_decay=1e-4
scheduler  = ReduceLROnPlateau, factor=0.5, patience=3
sampler    = WeightedRandomSampler
```

Macro F1 ceiling for this model and these hyperparameters: 0.888.

---

## Where Hyperparameter Tuning Ends

Macro F1 moved within a 0.851-0.888 band across all 7 runs. No configuration
broke through 0.89. The three weakest classes across every run:

```
Scratch:   F1 0.576 - 0.803  (high variance across runs)
Edge-Loc:  F1 0.771 - 0.830
Loc:       F1 0.708 - 0.772  (never exceeded 0.80 in any run)
```

Hyperparameter tuning is the right tool when the model has the right
information but is not using it optimally. Architecture changes are the
right tool when the model is missing information entirely. The ceiling
observed here indicates the latter; the next step is a model change,
not another hyperparameter experiment.

---

## References

Linear scaling rule for batch size and LR:
Goyal et al. 2017. Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.
https://arxiv.org/abs/1706.02677

Implicit regularization of small batch training:
Keskar et al. 2017. On Large-Batch Training for Deep Learning: Generalization
Gap and Sharp Minima.
https://arxiv.org/abs/1609.04836
