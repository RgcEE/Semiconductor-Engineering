# Semiconductor-Engineering

Author: Reynaldo Gomez
Semiconductor-Engineering

This repository is dedicated to semiconductor engineering work.

---

## Yield CNN

A multiclass CNN classifier for semiconductor wafer defect pattern recognition using the LSWMD dataset. Trains a 4-block convolutional network across 9 defect classes (8 defect types + none) and saves the best checkpoint based on validation loss.

Reference notes written during development. Written to be understood, not just referenced; each doc explains the why behind the implementation, not just what the code does.

### Documentation

#### [01_cnn_fundamentals.md](01_cnn_fundamentals.md)
What a CNN is mathematically. The forward pass layer by layer: convolution as a dot product, BatchNorm, SiLU vs ReLU, spatial downsampling, AdaptiveAvgPool, dropout, the linear classifier. The skip connection explained via gradient flow. Read this first.

#### [02_training_mechanics.md](02_training_mechanics.md)
The five-line training loop. CrossEntropyLoss vs FocalLoss with the math. Backpropagation and the chain rule. AdamW update equation explained term by term. Learning rate schedulers: ReduceLROnPlateau vs CosineAnnealingLR vs warm restarts. WeightedRandomSampler and why it is not the same as loss class weights. What overfitting looks like in the epoch output.

#### [03_reading_results.md](03_reading_results.md)
How to read the classification report column by column. The difference between precision, recall, F1, accuracy, macro avg, and weighted avg. Why accuracy is misleading for LSWMD. Statistical uncertainty per class based on sample count. How to compare experiments. The confusion matrix and what to look for. Why val loss is saved over accuracy.

#### [04_dynamic_training.md](04_dynamic_training.md)
The "differential score analyzer" concept mapped to real techniques: dynamic class weighting, learning rate warm restarts (SGDR), checkpoint-and-branch. How to open the training black box: per-batch loss logging, gradient norm monitoring, activation statistics, Grad-CAM. Research papers for each technique with direct links.

#### [05_batch_size_ablation.md](05_batch_size_ablation.md)
Batch size and learning rate ablation across seven runs. The mechanism behind batch=128 as the optimal configuration: implicit regularization, BatchNorm stability, and the linear scaling rule. The macro F1 ceiling at 0.888 interpreted as a representational capacity limit, not a hyperparameter problem.

#### [06_se_coord.md](06_se_coord.md)
SE attention and CoordConv ablation. Three implementation bugs documented with their effect on result validity traced run by run. Root cause of CoordConv's F1 regression derived. Decision to adopt SE attention as the new base architecture recorded with supporting experimental evidence.

---

### Experiment results

| Experiment | Epochs | Macro F1 | Donut F1 | Scratch F1 | Notes |
|---|---|---|---|---|---|
| Baseline (plain CNN) | 20 | 0.800 | 0.56 | 0.76 | CrossEntropyLoss, MaxPool |
| ResNet+Focal | 20 | 0.870 | 0.89 | 0.71 | ResNet + SiLU + FocalLoss |
| ResNet+Focal | 40 | 0.890 | 0.87 | 0.80 | +20 epochs, CosineAnnealingLR |
| Batch ablation best | 40 | 0.888 | 0.867 | 0.803 | batch=128, LR=3e-4, confirmed optimal |
| SE only | 40 | 0.886 | 0.873 | 0.802 | SE attention, ReduceLROnPlateau |

---

### Key references

He et al., 2015. *Deep Residual Learning for Image Recognition.*
https://arxiv.org/abs/1512.03385

Ioffe & Szegedy, 2015. *Batch Normalization: Accelerating Deep Network Training.*
https://arxiv.org/abs/1502.03167

Ramachandran et al., 2017. *Searching for Activation Functions.*
https://arxiv.org/abs/1710.05941

Lin et al., 2017. *Focal Loss for Dense Object Detection.*
https://arxiv.org/abs/1708.02002

Loshchilov & Hutter, 2017. *Decoupled Weight Decay Regularization.*
https://arxiv.org/abs/1711.05101

Loshchilov & Hutter, 2016. *SGDR: Stochastic Gradient Descent with Warm Restarts.*
https://arxiv.org/abs/1608.03983

Selvaraju et al., 2016. *Grad-CAM: Visual Explanations from Deep Networks.*
https://arxiv.org/abs/1610.02391

Goyal et al., 2017. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.*
https://arxiv.org/abs/1706.02677

Keskar et al., 2017. *On Large-Batch Training for Deep Learning.*
https://arxiv.org/abs/1609.04836

Cui et al., 2019. *Class-Balanced Loss Based on Effective Number of Samples.*
https://arxiv.org/abs/1901.05555

Shrivastava et al., 2016. *Training Region-based Object Detectors with Online Hard Example Mining.*
https://arxiv.org/abs/1604.03540

Shu et al., 2019. *Meta-Weight-Net: Learning an Explicit Mapping for Sample Weighting.*
https://arxiv.org/abs/1902.07379

Jaderberg et al., 2017. *Population Based Training of Neural Networks.*
https://arxiv.org/abs/1711.09846

Hu et al., 2018. *Squeeze-and-Excitation Networks.*
https://arxiv.org/abs/1709.01507

Liu et al., 2018. *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution.*
https://arxiv.org/abs/1807.03247

Goodfellow, Bengio, Courville. *Deep Learning.* MIT Press.
https://www.deeplearningbook.org/

PyTorch documentation.
https://pytorch.org/docs/stable/index.html