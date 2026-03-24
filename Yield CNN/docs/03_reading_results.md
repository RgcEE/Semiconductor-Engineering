# Reading results

Author: Reynaldo Gomez
Semiconductor-Engineering: Yield CNN

---

## The classification report

```
              precision    recall  f1-score   support

      Center       0.90      0.95      0.93       859
       Donut       0.85      0.88      0.87       111
    Edge-Loc       0.75      0.89      0.81      1038
   Edge-Ring       0.96      0.98      0.97      1936
         Loc       0.74      0.81      0.77       718
   Near-full       0.94      0.97      0.95        30
      Random       0.89      0.90      0.90       173
     Scratch       0.79      0.82      0.80       239
        none       0.99      0.98      0.99     29486

    accuracy                           0.97     34590
   macro avg       0.87      0.91      0.89     34590
weighted avg       0.97      0.97      0.97     34590
```

---

## What each column means

**Support** is the count of validation samples belonging to each class, the ground
truth population from which all other statistics are computed. This column should be
read first, as it determines how much weight to place on the reported metrics. Edge-Ring
has 1,936 samples; its F1 of 0.97 is a reliable estimate. Near-full has 30 samples;
its F1 of 0.95 carries a 95% confidence interval of ±0.076 (see the statistical
uncertainty section below). Donut at 111 samples has a 95% CI of ±0.063.

**Precision** is the fraction of the model's positive predictions for a given class
that are correct:

$$
\text{precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

Low precision means the model over-calls the class: it predicts it when it should not.
Scratch precision at 0.79 means 21% of wafers called Scratch were actually something
else, likely Edge-Loc or Loc, which occupy similar spatial zones. In a manufacturing
context, low precision wastes engineer review time on wafers that are not defective
with the predicted pattern.

**Recall** is the fraction of all ground-truth instances of a class that the model
correctly identifies:

$$
\text{recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

Low recall means the model misses real defects, predicting them as a different class.
If Donut recall were 0.40, 60% of actual Donut wafers would be miscategorized and
potentially allowed to continue through the process. Low recall is the more dangerous
failure mode: a false alarm wastes time, but a missed defect causes downstream yield
loss.

**F1-score** is the harmonic mean of precision and recall:

$$
F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

The harmonic mean penalizes imbalance between the two components more severely than
the arithmetic mean would. A model with precision=1.0 and recall=0.01 achieves
F1=0.02, not 0.50; the near-zero recall collapses the score. F1 cannot be gamed by
optimizing one metric at the expense of the other. In practice, F1 above 0.85 indicates
reliable performance; below 0.70 indicates the model is not learning the class
consistently enough for production use.

---

## The summary rows

**Accuracy** is the fraction of all predictions that are correct across the entire
validation set. On LSWMD it is not a useful metric. A classifier that predicts *none*
for every single wafer, never identifying any defect, would achieve 85.2% accuracy
(29,486/34,590), a higher number than many intermediate experiments produce. Accuracy
is algebraically dominated by the majority class and should not be used as the headline
metric for this problem.

**Macro F1** averages the per-class F1 scores treating every class with equal weight
regardless of its support count:

$$
\overline{F_1} = \frac{1}{9}\sum_{c} F_{1,c}
$$

This is the correct optimization target. Near-full (30 samples) and *none* (29,486
samples) contribute equally to macro F1; an improvement in macro F1 means the model
improved on the rare classes, not just on the majority. If only the weighted F1 improves,
the model refined its *none* performance while rare-class performance stagnated.

Macro F1 improved from 0.80 for the plain CNN baseline to 0.87 for ResNet+Focal at
20 epochs and 0.89 at 40 epochs, directly reflecting improvement on the hard classes
rather than the numerically dominant *none* class.

**Weighted F1** averages per-class F1 weighted by support count. Because *none*
accounts for 85.2% of the validation set, weighted F1 is nearly identical to accuracy
and provides essentially no additional information about rare-class performance. It
should not inform experimental decisions.

---

## Experiment-to-experiment comparison

The table below compares per-class F1 across the three main model versions:

```
Class       Baseline   20ep ResNet   40ep ResNet   Change (20->40)
---------   --------   -----------   -----------   ---------------
Center          0.88          0.91          0.93        +0.02
Donut           0.56          0.89          0.87        -0.02  (within uncertainty)
Edge-Loc        0.76          0.78          0.81        +0.03
Edge-Ring       0.97          0.97          0.97         0.00
Loc             0.68          0.75          0.77        +0.02
Near-full       0.85          0.93          0.95        +0.02
Random          0.76          0.90          0.90         0.00
Scratch         0.76          0.71          0.80        +0.09
none            0.99          0.98          0.99        +0.01

macro F1        0.80          0.87          0.89        +0.02
```

The most significant observation in the 20-to-40 epoch transition is Scratch recovering
from 0.71 to 0.80. What appeared at 20 epochs to be a regression from the baseline was
instead an indication that the model required additional training time to learn the
Scratch pattern under FocalLoss, whose gradient weighting shifts representation learning
toward hard examples that may require more optimization steps to converge. The Donut
drop of 0.02 from 20 to 40 epochs falls within the statistical uncertainty interval
for a class with 111 samples and does not represent a real regression.

---

## Statistical uncertainty in the metrics

F1 scores are point estimates. Their uncertainty depends on the number of samples in
the class, and small sample counts produce wide confidence intervals. Treating the F1
score as a proportion `p` estimated from `n` samples, the standard error is:

$$
\text{SE} = \sqrt{\frac{p(1-p)}{n}}
$$

Applying this to the reported classes (95% confidence interval = 1.96 × SE):

```
none      (n=29486, F1=0.99):  SE = 0.001  ->  0.99 +/- 0.002
Edge-Ring (n=1936,  F1=0.97):  SE = 0.004  ->  0.97 +/- 0.008
Donut     (n=111,   F1=0.87):  SE = 0.032  ->  0.87 +/- 0.063
Near-full (n=30,    F1=0.95):  SE = 0.039  ->  0.95 +/- 0.076
```

A 0.02 change in Donut F1 between two experiments is indistinguishable from measurement
noise given the 0.063 half-width. A 0.33 change, specifically the Donut improvement from 0.56 to
0.89 with the introduction of FocalLoss, is unambiguously real. A 0.02 change in *none*
F1 at n=29,486 is also a real effect, small as it appears. These intervals are also
the reason that increasing the Donut and Near-full sample counts through synthetic
augmentation or additional data collection matters not just for training signal but for
the statistical reliability of the evaluation metrics themselves.

---

## The confusion matrix

The confusion matrix provides the full breakdown of predicted class versus true class.
Running `yield_evaluate.py` generates it as a normalized PNG heatmap. Rows correspond
to the true class; columns to the predicted class. Diagonal entries are the per-class
recall values. Off-diagonal cell (i, j) is the fraction of true-class-i samples
predicted as class j.

The cells to focus on are the high-value off-diagonal pairs. A high value in
(Scratch, Edge-Loc), meaning the model predicts Edge-Loc for real Scratch wafers,
is consistent with the spatial ambiguity explanation: a scratch line that crosses the
edge zone activates features shared with edge-localized cluster defects, and the model
without explicit spatial position information cannot resolve the ambiguity. This is the
specific confusion pair that motivated the CoordConv experiments documented in
`06_se_coord.md`. A column that is bright everywhere indicates over-prediction of that
class; a row that is dim on the diagonal indicates low recall and systematic
misclassification to one or more other classes.

---

## Val loss vs accuracy as optimization targets

Checkpoints are saved on validation loss improvement, not on accuracy or macro F1
improvement. Validation loss is the direct output of the loss function. It is differentiable,
computed continuously over all nine classes simultaneously, and proportional to the
log-probability the model assigns to the correct answer. It is the signal that
backpropagation is designed to minimize, and it provides gradient-level precision about
model improvement.

Accuracy is a step function over the prediction: each sample is either correct or
incorrect. It is non-differentiable and dominated by the *none* class. F1 is also
non-differentiable and requires a full inference pass over the validation set with
`argmax` decoding to compute. Validation loss can be computed within the same forward
pass as the training step and carries richer signal.

In practice, validation loss improvement and macro F1 improvement are correlated but
not identical. The lowest validation loss observed in the batch ablation experiments
(0.2520) came from the configuration with the worst macro F1 (0.851), because a model
that trades slight *none* precision for rare-class improvement will show higher val loss
while achieving better macro F1. For this reason, both metrics are logged in `tracker.py`
and should be inspected together when comparing experiments.