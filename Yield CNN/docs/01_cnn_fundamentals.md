# CNN fundamentals

Author: Reynaldo Gomez
Semiconductor-Engineering: Yield CNN

---

## What a CNN is

A CNN is a mathematical function with learnable parameters.

$$
f_{\theta}(x) = \hat{y}
$$

- `x` is the input wafer map: a 64×64 grid of numbers
- $\hat{y}$ is the output: a vector of 9 scores, one per defect class
- $\theta$ is every learnable number in the network (weights and biases)

Training is finding the $\theta$ that makes $f_\theta$ good at its job. Every line of code in `yield_multi_classifier.py` and `yield_resnet_focal.py` exists to serve this one goal.

---

## The forward pass: What happens to the data

When `logits = model(x)` is called, the wafer map is transformed by a sequence of
differentiable operations before the final class scores are produced. Each stage
operates on the output of the previous one.

### Stage 1: Convolution

A 64×64 wafer map enters the first convolutional layer, which applies 32 learned filters
each of spatial extent 3×3. For each filter, the operation at every output position is
a dot product between the filter weights and the corresponding input patch, plus a scalar
bias:

$$
Z[i,j] = \sum_{m,n} W[m,n] \cdot x[i+m,\, j+n] + b
$$

`W` is the 3×3 weight matrix, `x` is the local input patch, and `b` is the bias. With
32 filters applied to a 64×64 input, the output is a (32, 64, 64) tensor: 32 feature
maps, each encoding where in the image one particular learned pattern was detected.

The filters are updated by gradient descent alongside all other parameters. In practice, shallow layers converge toward detecting low-level spatial structure, such as edges, gradients, and localized intensity blobs, while deeper layers combine these into curves, corners, rings, and full defect signatures. This hierarchical composition is the structural reason for stacking layers: each layer operates on the abstracted representations produced by all previous layers. In matrix form across all filters simultaneously, convolution is `Z = W * X + b`, a sequence of matrix multiplications separated by nonlinearities.

### Stage 2: Batch normalization

After each convolutional layer, BatchNorm2d normalizes the pre-activation values across
the spatial dimensions of the current mini-batch:

$$
\hat{x}_i = \frac{x_i - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}} \qquad y_i = \gamma \hat{x}_i + \beta
$$

`gamma` and `beta` are learnable parameters initialized to 1 and 0 respectively.
`epsilon` (typically 1e-5) prevents division by zero near-zero variance. Without normalization, activations across layers drift toward saturated or near-zero values as training depth increases: large activations push the nonlinearity into its flat region, zeroing the gradient; small activations produce no gradient signal and stop learning. BatchNorm keeps each layer's input distribution centered and scaled regardless of what the upstream layers are doing.

For LSWMD specifically, the 29,486 *none* samples and the 30 *Near-full* samples carry
very different statistical profiles. Without BatchNorm, the dominant class monopolizes
the activation range and rare classes receive negligible gradient signal. BatchNorm
normalizes within each batch across whichever samples happen to appear, preventing the
majority class from saturating the activations.

### Stage 3: Activation function

The nonlinearity inserted after each normalization step is what makes the network
capable of learning non-linear decision boundaries. Without a nonlinearity, any stack
of linear layers collapses to a single linear transformation regardless of depth, and
the network cannot represent the curved boundaries separating defect classes that occupy
overlapping regions of the feature space.

The baseline model uses ReLU, defined as `ReLU(x) = max(0, x)`. Its derivative is 1
for positive inputs and 0 for negative ones. Neurons that accumulate negative
pre-activations receive zero gradient on every batch and never update, exhibiting the dying
neuron pathology.

The ResNet model uses SiLU (also called Swish), defined as `SiLU(x) = x * sigmoid(x)`.
Its derivative is:

$$
\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))
$$

SiLU is smooth and nonzero across essentially its entire domain. Small negative values
still pass through with a small but non-zero gradient, giving the optimizer signal in
early training when a large fraction of pre-activations are negative and ReLU would
produce no learning. The practical effect is faster initial convergence and more robust
gradient flow in shallow layers.

### Stage 4: Spatial downsampling

Spatial downsampling reduces the feature map resolution so that later layers operate
on representations covering larger receptive fields of the original input. A 3×3
filter on a 4×4 feature map covers a region of the original 64×64 image that no single
filter in the first layer can see.

The baseline model uses `MaxPool2d(2)`, which takes the maximum value in each 2×2
non-overlapping window, reducing a (C, 64, 64) tensor to (C, 32, 32) while discarding
75% of the values. The ResNet model instead uses stride=2 in the convolutional layer
itself, stepping the filter by 2 positions at each update. This learns the downsampling
function from data rather than hardcoding it as a max operation, generally preserving
more information about thin spatial structures such as scratch lines that MaxPool tends
to suppress.

### Stage 5: AdaptiveAvgPool2d

At the end of the feature extractor, regardless of the spatial dimensions accumulated
through the earlier stages, `AdaptiveAvgPool2d(1)` collapses every feature map to a
single scalar by averaging all spatial positions. A (256, 4, 4) tensor becomes a
(256, 1, 1) tensor: 256 numbers, one per channel. The ``adaptive'' qualifier means
the target output size is specified rather than the pooling window, so the operation
produces the same output shape for any input resolution. This makes the model
forward-compatible with a change in `IMG_SIZE` without any modification to the
classifier head.

### Stage 6: Dropout

```python
self.dropout = nn.Dropout(0.4)
```

During the training forward pass, Dropout randomly zeroes 40% of the 256 post-pool
feature values before they reach the linear classifier. This prevents any single feature
from being relied upon exclusively. If a feature achieves high discriminative accuracy
for one class, the model is forced to learn redundant representations of the same
information across multiple features because any one of them may be suppressed on a
given forward pass. During evaluation (`model.eval()`), Dropout is disabled and all 256
features are passed through unchanged. The 0.4 rate represents a tuning decision:
higher rates add more regularization at the cost of slower convergence; lower rates
reduce regularization while allowing the model more expressive capacity.

### Stage 7: Linear classifier

$$
\text{logits} = W_{\text{cls}} \cdot \text{features} + b_{\text{cls}}
$$

A single matrix multiplication maps the 256-dimensional feature vector to the 9 class
scores. The output, shape (N, 9) for a batch of N samples, consists of raw logits,
not probabilities. Converting to probabilities requires softmax: `p = exp(logit) / sum(exp(logits))`.
`CrossEntropyLoss` applies the log-softmax internally; adding softmax to the model
definition would double-apply it and corrupt the loss computation.

---

## The skip connection

The residual block implements:

```python
def forward(self, x):
    identity = self.skip(x)
    out = self.act(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    return self.act(out + identity)
```

The block learns the residual `F(x) = desired_output - x` rather than the full mapping
`desired_output` directly. The identity shortcut `self.skip(x)`, a 1×1 convolution
when channel dimensions change or a pass-through when they match, is added to the
output before the final activation.

The gradient consequence is the essential point. During backpropagation, the gradient
of the loss with respect to parameters in an early layer `l` is:

$$
\frac{\partial L}{\partial \theta_l} = \frac{\partial L}{\partial z_L} \cdot \left(1 + \frac{\partial F(x)}{\partial x}\right)
$$

The `+1` term comes from the identity path and is constant regardless of what `F(x)`
is doing. In a plain network, the gradient is a product of Jacobians across every
layer; if each is slightly less than 1 in magnitude, the product approaches zero
exponentially with depth and early layers receive no useful signal. The skip connection
guarantees that the gradient through any layer is bounded from below by the direct
path through the identity, preventing the vanishing gradient regime that makes plain
deep networks difficult to train.

---

## References

He et al., 2015. *Deep Residual Learning for Image Recognition.*
https://arxiv.org/abs/1512.03385

Ioffe & Szegedy, 2015. *Batch Normalization: Accelerating Deep Network Training.*
https://arxiv.org/abs/1502.03167

Ramachandran et al., 2017. *Searching for Activation Functions.*
https://arxiv.org/abs/1710.05941

Goodfellow, Bengio, Courville. *Deep Learning.* MIT Press.
https://www.deeplearningbook.org/