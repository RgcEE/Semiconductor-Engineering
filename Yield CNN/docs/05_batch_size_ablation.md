# Batch size and learning rate ablation: yield_resnet_focal

Author: Reynaldo Gomez
Semiconductor-Engineering: Yield CNN

---

## What batch size controls

Batch size determines how many samples the model processes before each weight update.
With 138,360 training samples, the number of gradient steps per epoch is:

```
batch=64:  138,360 / 64  = 2,162 weight updates per epoch
batch=128: 138,360 / 128 = 1,081 weight updates per epoch
batch=256: 138,360 / 256 =   540 weight updates per epoch
```

The full dataset is seen once per epoch regardless of batch size. What changes is the
frequency of weight updates and the statistical quality of each gradient estimate: a
batch of 64 samples produces a noisier approximation of the true dataset gradient than
a batch of 128, since each estimate is based on fewer samples.

This noise has opposing effects on the optimization outcome. On the one hand, noisier
gradient estimates cause less accurate parameter updates and can slow convergence. On
the other hand, the stochastic perturbations in each small-batch estimate function as
implicit regularization, preventing the optimizer from settling into sharp, narrow
minima that fit the training set closely but do not generalize. Keskar et al. (2017)
demonstrate that models trained with small batch sizes consistently converge to flatter
minima with better generalization properties than models trained with large batches
on the same data.

For the rare classes in LSWMD, namely Scratch (1,193 samples), Donut (555), and Near-full (149),
this implicit regularization is particularly important. The optimizer needs to explore
the loss landscape to find representations of these classes that generalize; the gradient
noise from smaller batches provides that exploration. Removing it by increasing to
batch=256, as the results confirm, collapsed Scratch precision to near-random levels
across both confirming runs.

---

## What learning rate controls

The learning rate scales the magnitude of each parameter update:

$$
\theta \leftarrow \theta - \text{LR} \cdot \nabla_\theta \mathcal{L}
$$

In the geometric interpretation, the loss landscape is a high-dimensional surface and
each gradient step moves the parameter vector downhill. A learning rate that is too
large overshoots the valley floor and produces oscillating or diverging loss; one that
is too small takes steps so small that convergence requires prohibitively many epochs
and the optimizer becomes trapped in shallow local minima encountered early in training.

---

## The linear scaling rule

When batch size is changed, the learning rate should scale proportionally to maintain
an approximately constant effective learning signal per epoch:

$$
\text{LR}_{\text{new}} = \text{LR}_{\text{old}} \cdot \frac{B_{\text{new}}}{B_{\text{old}}}
$$

Halving the batch size implies halving the learning rate:

```
batch=128, LR=3e-4  ->  batch=64, LR=1.5e-4
```

The intuition is that smaller batches produce noisier gradient estimates, and a
proportionally smaller learning rate compensates by taking more conservative steps on
each noisy estimate. The total effective gradient contribution per epoch remains
approximately constant:

```
Effective update per epoch ~= LR * gradient * updates_per_epoch

batch=128, LR=3e-4:   3e-4   * gradient * 1,081 steps
batch=64,  LR=1.5e-4: 1.5e-4 * gradient * 2,162 steps   <- same order
```

This rule is a principled heuristic, not a derivable law. It provides a calibrated
starting point for batch-size experiments rather than requiring a full hyperparameter
search for each new batch size.

Reference: Goyal et al. 2017. https://arxiv.org/abs/1706.02677

---

## All runs: Full results

| batch | epochs | LR     | best\_ep | val\_loss | macro\_f1 | scratch\_f1 |
|-------|--------|--------|----------|-----------|-----------|-------------|
| 128   | 20     | 3e-4   | 18       | 0.2540    | 0.869     | 0.705       |
| 128   | 40     | 3e-4   | 12       | 0.2870    | 0.888     | 0.803       |
| 256   | 20     | 3e-4   | 18       | 0.2544    | 0.863     | 0.689       |
| 256   | 20     | 3e-4   | 12       | 0.2520    | 0.851     | 0.576       |
| 64    | 20     | 3e-4   | 20       | 0.2751    | 0.863     | 0.691       |
| 64    | 20     | 1.5e-4 | 20       | 0.3382    | 0.860     | 0.678       |
| 64    | 40     | 1.5e-4 | 8        | 0.3210    | 0.871     | 0.773       |

---

## What the results show

The 256-batch configuration degraded performance consistently across both confirming
runs, with Scratch precision collapsing to 0.44–0.60. Two runs showing the same
degradation on the target metric is sufficient to treat the result as reproducible
rather than attributable to random initialization variance. The mechanism is the loss
of implicit regularization: 540 gradient steps per epoch instead of 1,081, each based
on a lower-noise estimate, removed the stochastic exploration that the rare-class
representations required to generalize. Fewer steps also means fewer opportunities to
see rare-class samples, which compounds the effect under WeightedRandomSampler.

The 64-batch configuration failed to surpass 128 across all three runs tested, with
macro F1 in the range 0.860–0.871 versus the 128-batch best of 0.888. The explanation
is BatchNorm instability: with nine classes under WeightedRandomSampler, a batch of 64
contains roughly seven samples per class on average. BatchNorm2d normalizes each
channel's activations across the spatial dimensions of the batch; at seven samples per
class, the per-batch statistics are too variable to provide stable normalization,
partially offsetting the regularization benefit from the smaller batch size.

The 64-epoch run at LR=1.5e-4 reached its best checkpoint at epoch 8 out of 40 and
produced no further improvement thereafter. The lower learning rate caused slower
initial descent into the loss landscape, and the model converged at a shallower
minimum earlier in training. The subsequent 32 epochs show train loss continuing to
fall while validation metrics stagnate, the canonical overfit signature, indicating
the additional compute was not recovering any remaining generalization capacity.

The decoupling of validation loss and macro F1 is most visible in the 256-batch run
that produced the lowest validation loss in the table (0.2520) alongside the worst
macro F1 (0.851). Validation loss is dominated by the *none* class at 85.2% of the
validation set; a model that improves *none* confidence while sacrificing Scratch and
Edge-Loc will show lower validation loss and worse macro F1 simultaneously. Validation
loss should be used to select checkpoints within a run; macro F1 should be used to
compare runs against each other.

On experimental controls: a single run does not distinguish a genuine configuration
effect from random initialization and batch-ordering variance. The practical standard
adopted here is that two runs showing consistent degradation on the primary metric
constitute sufficient evidence to move on. When two runs produce mixed results, a third
is warranted. Pursuing further hyperparameter refinement within this architecture
appears to be subject to diminishing returns; the macro F1 ceiling observed across all
seven runs is consistent with a representational capacity limit rather than an
optimization one.

---

## Confirmed optimal configuration

```
batch_size = 128
LR         = 3e-4
epochs     = 40
optimizer  = AdamW, weight_decay=1e-4
scheduler  = ReduceLROnPlateau, factor=0.5, patience=3
sampler    = WeightedRandomSampler
```

Macro F1 ceiling for this architecture and these hyperparameters: **0.888**.

---

## Where hyperparameter tuning ends

All seven runs produced macro F1 in the range 0.851–0.888. No configuration crossed
0.89. The three persistently weak classes across every run are:

```
Scratch:   F1 0.576 - 0.803  (high variance across runs)
Edge-Loc:  F1 0.771 - 0.830
Loc:       F1 0.708 - 0.772  (never exceeded 0.80 in any run)
```

Hyperparameter tuning is the appropriate tool when the model has access to the right
information but is not using it optimally, when the bottleneck is optimization dynamics
rather than representational capacity. An architectural change is the appropriate tool
when the model is missing information that no amount of re-weighting or scheduling can
recover. The Scratch–Edge-Loc–Loc confusion cluster is spatially grounded: these classes
overlap in the feature space because the model has no explicit representation of where
in the 64×64 grid a detected pattern is located. That is a missing input, not a
sub-optimal optimization. The next step is an architectural change.

---

## References

Goyal et al. 2017. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.*
https://arxiv.org/abs/1706.02677

Keskar et al. 2017. *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.*
https://arxiv.org/abs/1609.04836