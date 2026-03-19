# CNN Fundamentals
**Author: Reynaldo Gomez**
**Repo: Semiconductor-Engineering / Yield CNN**

---

## What a CNN Is

A CNN is a mathematical function with learnable parameters.

```
f_theta(x) = y_hat
```

- `x` is the input wafer map: a 64x64 grid of numbers
- `y_hat` is the output: a vector of 9 scores, one per defect class
- `theta` is every learnable number in the network (weights and biases)

Training is finding theta that makes f_theta good at its job.
Every line of code in yield_multi_classifier.py and yield_resnet_focal.py
exists to serve this one goal.

---

## The Forward Pass: What Happens to the Data

When `logits = model(x)` is called, the wafer map flows through these stages
in order. Each stage transforms the data before passing it to the next.

### Stage 1: Convolution

A 64x64 wafer map enters. The first conv layer has 32 filters, each 3x3.
Each filter slides across the image and computes a dot product at every position:

```
output[i,j] = sum over m,n of: W[m,n] * x[i+m, j+n]  +  b
```

W is the 3x3 filter weight matrix. x is the input patch. b is a bias scalar.
This produces one number per position per filter.

With 32 filters on a 64x64 input the output is (32, 64, 64): 32 feature maps,
each showing where that filter's pattern was detected across the image.

What the filters learn to detect:
- Layer 1 filters: edges, gradients, blobs
- Layer 2 filters: combinations of edges (curves, corners, short lines)
- Layer 3 filters: combinations of curves (rings, clusters, partial patterns)
- Layer 4 filters: full defect signatures (complete rings, scratch lines, center blobs)

This hierarchical feature building is why layers stack. Each layer sees
what the previous layer found and builds on it.

In matrix form, convolution across all filters is:
```
Z = W * X + b
```
The entire forward pass is a sequence of these matrix multiplies with
nonlinearities inserted between them.

### Stage 2: Batch Normalization

After each conv, BatchNorm2d normalizes the activations:

```
x_hat_i = (x_i - mean_batch) / sqrt(variance_batch + epsilon)
y_i = gamma * x_hat_i + beta
```

gamma and beta are learnable parameters. epsilon (typically 1e-5) prevents
division by zero.

Without BatchNorm, activations across layers drift toward very large or very
small values as training progresses. Large activations saturate the nonlinearity
(output stops changing with input). Small activations produce near-zero gradients
(weights stop updating). BatchNorm keeps everything in a stable range.

For LSWMD specifically: the 29,486 none wafers and the 30 Near-full wafers have
very different statistical profiles. Without BatchNorm, the dominant class pushes
activations into saturated regions and the rare classes get no gradient signal.
BatchNorm normalizes across whichever samples appear in the batch, preventing
any class from monopolizing the activation range.

### Stage 3: Activation Function

The nonlinearity that makes the network able to learn complex patterns.
Without it, stacking linear layers is mathematically equivalent to one linear
layer; curved decision boundaries are impossible.

**ReLU (baseline model):**
```
ReLU(x) = max(0, x)
```
Zero gradient for x < 0. Neurons that activate negatively are dead to
the optimizer; they receive no gradient and never update.

**SiLU / Swish (resnet_focal model):**
```
SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
```
Derivative:
```
SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
```
Smooth and nonzero almost everywhere. Small negative values pass through
with a small gradient. Gives the optimizer more signal in early training
when many activations are negative.

### Stage 4: Spatial Downsampling

Reduces the spatial dimensions so later layers see larger regions of the input.

**Baseline (MaxPool2d(2)):** Takes the maximum value in each 2x2 window.
(64,64) becomes (32,32). Fast, but discards 75% of values.

**ResNet model (stride=2 in conv):** The conv itself uses a step size of 2,
so output positions are spaced 2 pixels apart on the input. Learns the
downsampling function rather than hardcoding max. Generally better at
preserving fine structure like thin scratch lines.

### Stage 5: AdaptiveAvgPool2d

At the end of the feature extractor, regardless of spatial size, this
collapses every feature map to a single number by averaging all values.

(256, 4, 4) becomes (256, 1, 1): 256 numbers total, one per channel.

The "adaptive" means the output size is specified, not the window size.
If IMG_SIZE changes from 64 to 128, AdaptiveAvgPool still produces (256, 1, 1).
The model handles variable input sizes without any code changes.

### Stage 6: Dropout

```python
self.dropout = nn.Dropout(0.4)
```

During training, randomly sets 40% of the 256 features to zero.
Forces the network to not rely on any single feature; if it does,
that feature gets zeroed randomly and the model learns to use redundant
representations.

During eval (model.eval()), dropout is disabled. All 256 features are active.

### Stage 7: Linear Classifier

```
logits = W_classifier * features + b_classifier
```

A matrix multiply from 256 features to K=9 class scores.
Output shape: (N, 9): 9 raw scores per sample.

These are called logits. They are NOT probabilities yet.
To get probabilities: softmax(logits).
CrossEntropyLoss applies softmax internally, so it never needs to be added
to the model.

---

## The Skip Connection

In ResidualBlock:

```python
def forward(self, x):
    identity = self.skip(x)
    out = self.act(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    return self.act(out + identity)
```

The block learns F(x) = desired_output - x (the residual).
The full output is F(x) + x.

During backpropagation, gradients flow through two paths simultaneously:
through the conv layers AND directly through the skip. The gradient at
an early layer is:

```
d_Loss/d_theta_l = d_Loss/d_z_L * (1 + d_F(x)/d_x)
```

The +1 term guarantees the gradient is never smaller than the direct path.
Plain CNNs multiply gradients through every layer; if each is slightly less
than 1, the product approaches zero and early layers stop learning.
Residual networks avoid this entirely.

---

## References

ResNet original paper (He et al., 2015):
https://arxiv.org/abs/1512.03385

BatchNorm original paper (Ioffe & Szegedy, 2015):
https://arxiv.org/abs/1502.03167

SiLU / Swish activation (Ramachandran et al., 2017):
https://arxiv.org/abs/1710.05941

Deep Learning textbook (Goodfellow et al.), free online:
https://www.deeplearningbook.org/
