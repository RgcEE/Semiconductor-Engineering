//Name: Reynaldo Gomez
//Last Edited: 2/4/2026
//Reference: Advanced Cuda C++ Algorithms for Semiconductor Engineering - Xi Shan

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

//---------------------------------------------------
// Constants and Simple Structures
//---------------------------------------------------

// CNN Configuration (demo)
constexpr int BATCH_SIZE    = 8;  // each block processes one mini-batch
constexpr int INPUT_WIDTH   = 8;
constexpr int INPUT_HEIGHT  = 8;
constexpr int FILTER_SIZE   = 3;
constexpr int NUM_FILTERS   = 1;
constexpr int FC_OUT        = 1;  // output dimension

// Derived dimensions
constexpr int CONV_OUT_WIDTH  = INPUT_WIDTH  - FILTER_SIZE + 1;
constexpr int CONV_OUT_HEIGHT = INPUT_HEIGHT - FILTER_SIZE + 1;

// Dataset/training
constexpr int NUM_SAMPLES = 32;
constexpr int EPOCHS      = 2;

//---------------------------------------------------
// CUDA error checking
//---------------------------------------------------
#define CUDA_CHECK(err)                                                      \
  do {                                                                       \
    cudaError_t _e = (err);                                                  \
    if (_e != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at line %d in file %s\n",               \
              cudaGetErrorString(_e), __LINE__, __FILE__);                   \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

//---------------------------------------------------
// Forward declarations (signatures must match definitions)
//---------------------------------------------------
__global__ void forwardConvKernel(
    const float* __restrict__ d_input,
    const float* __restrict__ d_filter,
    const float* __restrict__ d_biasConv,
    float* __restrict__ d_convOut,
    int batchOffset
);

__global__ void forwardFCKernel(
    const float* __restrict__ d_convOut,
    const float* __restrict__ d_fcWeight,
    const float* __restrict__ d_biasFC,
    float* __restrict__ d_fcOut,
    int batchOffset
);

__global__ void computeLossAndDoutKernel(
    const float* __restrict__ d_fcOut,
    const float* __restrict__ d_labels,
    float* __restrict__ d_loss,
    float* __restrict__ d_fcOutGrad,
    int batchOffset
);

__global__ void backwardFCKernel(
    const float* __restrict__ d_convOut,
    const float* __restrict__ d_fcOutGrad,
    float* __restrict__ d_fcWeightGrad,
    float* __restrict__ d_biasFCGrad,
    int batchOffset
);

__global__ void backwardConvKernel(
    const float* __restrict__ d_input,
    const float* __restrict__ d_fcOutGrad,
    const float* __restrict__ d_fcWeight,
    float* __restrict__ d_filterGrad,
    float* __restrict__ d_biasConvGrad,
    int batchOffset
);

__global__ void aggregateGradientsKernel(
    float* __restrict__ d_fcWeightGrad,
    float* __restrict__ d_biasFCGrad,
    float* __restrict__ d_filterGrad,
    float* __restrict__ d_biasConvGrad
);

__global__ void updateWeightsKernel(
    float* __restrict__ d_filter,
    float* __restrict__ d_biasConv,
    float* __restrict__ d_fcWeight,
    float* __restrict__ d_biasFC,
    const float* __restrict__ d_filterGrad,
    const float* __restrict__ d_biasConvGrad,
    const float* __restrict__ d_fcWeightGrad,
    const float* __restrict__ d_biasFCGrad,
    float learningRate
);

//---------------------------------------------------
// Main: Training loop
//---------------------------------------------------
int main() {
  // -------------------------
  // Host allocations (sizes)
  // -------------------------
  const size_t inputBytes      = static_cast<size_t>(NUM_SAMPLES) * INPUT_WIDTH * INPUT_HEIGHT * sizeof(float);
  const size_t labelBytes      = static_cast<size_t>(NUM_SAMPLES) * sizeof(float); // scalar label per sample
  const size_t convOutBytes    = static_cast<size_t>(NUM_SAMPLES) * CONV_OUT_WIDTH * CONV_OUT_HEIGHT * sizeof(float);
  const size_t fcOutBytes      = static_cast<size_t>(NUM_SAMPLES) * FC_OUT * sizeof(float);

  const size_t filterBytes     = static_cast<size_t>(NUM_FILTERS) * FILTER_SIZE * FILTER_SIZE * sizeof(float);
  const size_t biasConvBytes   = static_cast<size_t>(NUM_FILTERS) * sizeof(float);

  const size_t flattenSize     = static_cast<size_t>(NUM_FILTERS) * CONV_OUT_WIDTH * CONV_OUT_HEIGHT;
  const size_t fcWeightBytes   = flattenSize * FC_OUT * sizeof(float);
  const size_t biasFCBytes     = static_cast<size_t>(FC_OUT) * sizeof(float);

  // Host data
  float* h_input  = new float[static_cast<size_t>(NUM_SAMPLES) * INPUT_WIDTH * INPUT_HEIGHT];
  float* h_labels = new float[NUM_SAMPLES];

  // Host gradient buffers (for final copy-back demo)
  float* h_filterGrad   = new float[NUM_FILTERS * FILTER_SIZE * FILTER_SIZE];
  float* h_biasConvGrad = new float[NUM_FILTERS];
  float* h_fcWeightGrad = new float[flattenSize * FC_OUT];
  float* h_biasFCGrad   = new float[FC_OUT];

  // Random init input + labels
  for (int n = 0; n < NUM_SAMPLES; ++n) {
    h_labels[n] = (rand() % 2 == 0) ? 1.0f : 0.0f;
    for (int i = 0; i < INPUT_WIDTH * INPUT_HEIGHT; ++i) {
      h_input[n * (INPUT_WIDTH * INPUT_HEIGHT) + i] =
          static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
  }

  // -------------------------
  // Device allocations
  // -------------------------
  float *d_input = nullptr, *d_labels = nullptr;
  float *d_filter = nullptr, *d_biasConv = nullptr;
  float *d_fcWeight = nullptr, *d_biasFC = nullptr;
  float *d_convOut = nullptr, *d_fcOut = nullptr;
  float *d_fcOutGrad = nullptr, *d_loss = nullptr;

  float *d_filterGrad = nullptr, *d_biasConvGrad = nullptr;
  float *d_fcWeightGrad = nullptr, *d_biasFCGrad = nullptr;

  CUDA_CHECK(cudaMalloc((void**)&d_input,     inputBytes));
  CUDA_CHECK(cudaMalloc((void**)&d_labels,    labelBytes));
  CUDA_CHECK(cudaMalloc((void**)&d_filter,    filterBytes));
  CUDA_CHECK(cudaMalloc((void**)&d_biasConv,  biasConvBytes));
  CUDA_CHECK(cudaMalloc((void**)&d_fcWeight,  fcWeightBytes));
  CUDA_CHECK(cudaMalloc((void**)&d_biasFC,    biasFCBytes));
  CUDA_CHECK(cudaMalloc((void**)&d_convOut,   convOutBytes));
  CUDA_CHECK(cudaMalloc((void**)&d_fcOut,     fcOutBytes));
  CUDA_CHECK(cudaMalloc((void**)&d_fcOutGrad, fcOutBytes));
  CUDA_CHECK(cudaMalloc((void**)&d_loss,      BATCH_SIZE * sizeof(float))); // per-block partial losses

  CUDA_CHECK(cudaMalloc((void**)&d_filterGrad,   filterBytes));
  CUDA_CHECK(cudaMalloc((void**)&d_biasConvGrad, biasConvBytes));
  CUDA_CHECK(cudaMalloc((void**)&d_fcWeightGrad, fcWeightBytes));
  CUDA_CHECK(cudaMalloc((void**)&d_biasFCGrad,   biasFCBytes));

  // Copy host data to device
  CUDA_CHECK(cudaMemcpy(d_input,  h_input,  inputBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_labels, h_labels, labelBytes, cudaMemcpyHostToDevice));

  // -------------------------
  // Initialize weights/biases (host then copy)
  // -------------------------
  float* h_filterInit   = new float[NUM_FILTERS * FILTER_SIZE * FILTER_SIZE];
  float* h_biasConvInit = new float[NUM_FILTERS];
  float* h_fcWeightInit = new float[flattenSize * FC_OUT];
  float* h_biasFCInit   = new float[FC_OUT];

  for (int i = 0; i < NUM_FILTERS * FILTER_SIZE * FILTER_SIZE; ++i) {
    h_filterInit[i] = 0.01f * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  }
  for (int i = 0; i < NUM_FILTERS; ++i) h_biasConvInit[i] = 0.0f;

  for (size_t i = 0; i < flattenSize * FC_OUT; ++i) {
    h_fcWeightInit[i] = 0.01f * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  }
  for (int i = 0; i < FC_OUT; ++i) h_biasFCInit[i] = 0.0f;

  CUDA_CHECK(cudaMemcpy(d_filter,   h_filterInit,   filterBytes,   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_biasConv, h_biasConvInit, biasConvBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_fcWeight, h_fcWeightInit, fcWeightBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_biasFC,   h_biasFCInit,   biasFCBytes,   cudaMemcpyHostToDevice));

  delete[] h_filterInit;
  delete[] h_biasConvInit;
  delete[] h_fcWeightInit;
  delete[] h_biasFCInit;

  // -------------------------
  // Training config
  // -------------------------
  const int numBlocks = (NUM_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE;
  const float lr = 0.1f;

  dim3 grid(numBlocks);
  dim3 block(32);

  std::cout << "Starting training for " << EPOCHS << " epochs\n";

  for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    // clear grads each epoch (this matches your “aggregate then update” demo)
    CUDA_CHECK(cudaMemset(d_filterGrad,   0, filterBytes));
    CUDA_CHECK(cudaMemset(d_biasConvGrad, 0, biasConvBytes));
    CUDA_CHECK(cudaMemset(d_fcWeightGrad, 0, fcWeightBytes));
    CUDA_CHECK(cudaMemset(d_biasFCGrad,   0, biasFCBytes));

    // 1) forward conv
    forwardConvKernel<<<grid, block, FILTER_SIZE * FILTER_SIZE * sizeof(float)>>>(
        d_input, d_filter, d_biasConv, d_convOut, /*batchOffset=*/0);

    // 2) forward FC
    forwardFCKernel<<<grid, block>>>(
        d_convOut, d_fcWeight, d_biasFC, d_fcOut, /*batchOffset=*/0);

    // 3) loss + dL/d(fcOut)
    computeLossAndDoutKernel<<<grid, block>>>(
        d_fcOut, d_labels, d_loss, d_fcOutGrad, /*batchOffset=*/0);

    // 4) backward FC (accumulate dW_fc, db_fc)
    backwardFCKernel<<<grid, block>>>(
        d_convOut, d_fcOutGrad, d_fcWeightGrad, d_biasFCGrad, /*batchOffset=*/0);

    // 5) backward conv (accumulate dFilter, db_conv)
    backwardConvKernel<<<grid, block>>>(
        d_input, d_fcOutGrad, d_fcWeight, d_filterGrad, d_biasConvGrad, /*batchOffset=*/0);

    // 6) cross-block sync point (no-op kernel, but forces completion)
    aggregateGradientsKernel<<<1, 1>>>(
        d_fcWeightGrad, d_biasFCGrad, d_filterGrad, d_biasConvGrad);

    // 7) update
    updateWeightsKernel<<<1, 1>>>(
        d_filter, d_biasConv, d_fcWeight, d_biasFC,
        d_filterGrad, d_biasConvGrad, d_fcWeightGrad, d_biasFCGrad,
        lr);

    // pull loss for a quick metric (only block 0’s d_loss is meaningful here;
    // this mirrors your original “demo” style, not a correct full-dataset mean)
    float h_lossBatch[BATCH_SIZE];
    CUDA_CHECK(cudaMemcpy(h_lossBatch, d_loss, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    float totalLoss = 0.0f;
    for (int i = 0; i < BATCH_SIZE; ++i) totalLoss += h_lossBatch[i];
    totalLoss /= static_cast<float>(BATCH_SIZE);

    std::cout << "Epoch " << epoch << " average batch loss = " << totalLoss << "\n";
  }

  // -------------------------
  // Copy final grads to host (demo)
  // -------------------------
  CUDA_CHECK(cudaMemcpy(h_filterGrad,   d_filterGrad,   filterBytes,   cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_biasConvGrad, d_biasConvGrad, biasConvBytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_fcWeightGrad, d_fcWeightGrad, fcWeightBytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_biasFCGrad,   d_biasFCGrad,   biasFCBytes,   cudaMemcpyDeviceToHost));

  std::cout << "Final filter gradient example: " << h_filterGrad[0] << "\n";

  // -------------------------
  // Cleanup
  // -------------------------
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_labels));
  CUDA_CHECK(cudaFree(d_filter));
  CUDA_CHECK(cudaFree(d_biasConv));
  CUDA_CHECK(cudaFree(d_fcWeight));
  CUDA_CHECK(cudaFree(d_biasFC));
  CUDA_CHECK(cudaFree(d_convOut));
  CUDA_CHECK(cudaFree(d_fcOut));
  CUDA_CHECK(cudaFree(d_fcOutGrad));
  CUDA_CHECK(cudaFree(d_loss));
  CUDA_CHECK(cudaFree(d_filterGrad));
  CUDA_CHECK(cudaFree(d_biasConvGrad));
  CUDA_CHECK(cudaFree(d_fcWeightGrad));
  CUDA_CHECK(cudaFree(d_biasFCGrad));

  delete[] h_input;
  delete[] h_labels;
  delete[] h_filterGrad;
  delete[] h_biasConvGrad;
  delete[] h_fcWeightGrad;
  delete[] h_biasFCGrad;

  std::cout << "Training completed.\n";
  return 0;
}

//---------------------------------------------------
// Kernel implementations
//---------------------------------------------------

// forwardConvKernel: each block processes BATCH_SIZE samples, 1 filter
__global__ void forwardConvKernel(
    const float* __restrict__ d_input,
    const float* __restrict__ d_filter,
    const float* __restrict__ d_biasConv,
    float* __restrict__ d_convOut,
    int /*batchOffset*/)
{
  const int batchIndex = blockIdx.x;
  const int numBatches = (NUM_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE;
  if (batchIndex >= numBatches) return;

  extern __shared__ float s_filter[];
  if (threadIdx.x < FILTER_SIZE * FILTER_SIZE) {
    s_filter[threadIdx.x] = d_filter[threadIdx.x];
  }
  __syncthreads();

  int t = threadIdx.x;
  const int outPlane = CONV_OUT_HEIGHT * CONV_OUT_WIDTH;

  while (t < BATCH_SIZE * outPlane) {
    const int localSample = t / outPlane;
    const int outIndex    = t % outPlane;

    const int sampleIdx = batchIndex * BATCH_SIZE + localSample;
    if (sampleIdx >= NUM_SAMPLES) {
      t += blockDim.x;
      continue;
    }

    const int outRow = outIndex / CONV_OUT_WIDTH;
    const int outCol = outIndex % CONV_OUT_WIDTH;

    float sumVal = 0.0f;
    for (int fr = 0; fr < FILTER_SIZE; ++fr) {
      for (int fc = 0; fc < FILTER_SIZE; ++fc) {
        const int inRow = outRow + fr;
        const int inCol = outCol + fc;
        const float fval = s_filter[fr * FILTER_SIZE + fc];
        const float ival = d_input[sampleIdx * (INPUT_HEIGHT * INPUT_WIDTH)
                                   + (inRow * INPUT_WIDTH + inCol)];
        sumVal += fval * ival;
      }
    }

    sumVal += d_biasConv[0];

    d_convOut[sampleIdx * outPlane + (outRow * CONV_OUT_WIDTH + outCol)] = sumVal;

    t += blockDim.x;
  }
}

// forwardFCKernel: flatten conv output and compute 1 FC output with sigmoid
__global__ void forwardFCKernel(
    const float* __restrict__ d_convOut,
    const float* __restrict__ d_fcWeight,
    const float* __restrict__ d_biasFC,
    float* __restrict__ d_fcOut,
    int /*batchOffset*/)
{
  const int batchIndex = blockIdx.x;
  const int numBatches = (NUM_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE;
  if (batchIndex >= numBatches) return;

  int t = threadIdx.x;
  const int flattenN = CONV_OUT_HEIGHT * CONV_OUT_WIDTH; // NUM_FILTERS=1 demo

  while (t < BATCH_SIZE) {
    const int sampleIdx = batchIndex * BATCH_SIZE + t;
    if (sampleIdx >= NUM_SAMPLES) {
      t += blockDim.x;
      continue;
    }

    float sumVal = 0.0f;
    for (int i = 0; i < flattenN; ++i) {
      const float x = d_convOut[sampleIdx * flattenN + i];
      const float w = d_fcWeight[i];
      sumVal += x * w;
    }
    sumVal += d_biasFC[0];

    const float outVal = 1.0f / (1.0f + expf(-sumVal));
    d_fcOut[sampleIdx] = outVal;

    t += blockDim.x;
  }
}

// computeLossAndDoutKernel: BCE loss + derivative w.r.t. pred
__global__ void computeLossAndDoutKernel(
    const float* __restrict__ d_fcOut,
    const float* __restrict__ d_labels,
    float* __restrict__ d_loss,
    float* __restrict__ d_fcOutGrad,
    int /*batchOffset*/)
{
  const int batchIndex = blockIdx.x;
  const int numBatches = (NUM_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE;
  if (batchIndex >= numBatches) return;

  int t = threadIdx.x;
  while (t < BATCH_SIZE) {
    const int sampleIdx = batchIndex * BATCH_SIZE + t;
    if (sampleIdx >= NUM_SAMPLES) {
      t += blockDim.x;
      continue;
    }

    const float pred  = d_fcOut[sampleIdx];
    const float label = d_labels[sampleIdx];

    const float eps = 1e-6f;
    const float lossVal = -(label * logf(pred + eps) +
                            (1.0f - label) * logf(1.0f - pred + eps));

    const float grad = (pred - label) / ((pred + eps) * (1.0f - pred + eps));

    d_fcOutGrad[sampleIdx] = grad;
    d_loss[t] = lossVal;

    t += blockDim.x;
  }
}

// backwardFCKernel: accumulate dW_fc and db_fc using atomics
__global__ void backwardFCKernel(
    const float* __restrict__ d_convOut,
    const float* __restrict__ d_fcOutGrad,
    float* __restrict__ d_fcWeightGrad,
    float* __restrict__ d_biasFCGrad,
    int /*batchOffset*/)
{
  const int batchIndex = blockIdx.x;
  const int numBatches = (NUM_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE;
  if (batchIndex >= numBatches) return;

  int t = threadIdx.x;
  const int flattenN = CONV_OUT_HEIGHT * CONV_OUT_WIDTH;

  while (t < BATCH_SIZE) {
    const int sampleIdx = batchIndex * BATCH_SIZE + t;
    if (sampleIdx >= NUM_SAMPLES) {
      t += blockDim.x;
      continue;
    }

    const float gradOut = d_fcOutGrad[sampleIdx];

    for (int i = 0; i < flattenN; ++i) {
      const float x = d_convOut[sampleIdx * flattenN + i];
      atomicAdd(&d_fcWeightGrad[i], x * gradOut);
    }
    atomicAdd(&d_biasFCGrad[0], gradOut);

    t += blockDim.x;
  }
}

// backwardConvKernel: accumulate dFilter and dbConv (chain through FC)
__global__ void backwardConvKernel(
    const float* __restrict__ d_input,
    const float* __restrict__ d_fcOutGrad,
    const float* __restrict__ d_fcWeight,
    float* __restrict__ d_filterGrad,
    float* __restrict__ d_biasConvGrad,
    int /*batchOffset*/)
{
  const int batchIndex = blockIdx.x;
  const int numBatches = (NUM_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE;
  if (batchIndex >= numBatches) return;

  int t = threadIdx.x;

  while (t < BATCH_SIZE * FILTER_SIZE * FILTER_SIZE) {
    const int localSample = t / (FILTER_SIZE * FILTER_SIZE);
    const int filterIdx   = t % (FILTER_SIZE * FILTER_SIZE);

    const int sampleIdx = batchIndex * BATCH_SIZE + localSample;
    if (sampleIdx >= NUM_SAMPLES) {
      t += blockDim.x;
      continue;
    }

    const int fr = filterIdx / FILTER_SIZE;
    const int fc = filterIdx % FILTER_SIZE;

    const float gradOutSample = d_fcOutGrad[sampleIdx];

    float accumGrad = 0.0f;
    for (int oc_r = 0; oc_r < CONV_OUT_HEIGHT; ++oc_r) {
      for (int oc_c = 0; oc_c < CONV_OUT_WIDTH; ++oc_c) {
        const int outIdx = oc_r * CONV_OUT_WIDTH + oc_c;
        const float wFC = d_fcWeight[outIdx];

        const int inRow = oc_r + fr;
        const int inCol = oc_c + fc;
        const float inVal = d_input[sampleIdx * (INPUT_HEIGHT * INPUT_WIDTH) +
                                    (inRow * INPUT_WIDTH + inCol)];

        accumGrad += gradOutSample * wFC * inVal;
      }
    }
    atomicAdd(&d_filterGrad[filterIdx], accumGrad);

    // conv bias grad
    float accumBias = 0.0f;
    for (int oc_r = 0; oc_r < CONV_OUT_HEIGHT; ++oc_r) {
      for (int oc_c = 0; oc_c < CONV_OUT_WIDTH; ++oc_c) {
        const float wFC = d_fcWeight[oc_r * CONV_OUT_WIDTH + oc_c];
        accumBias += gradOutSample * wFC;
      }
    }
    atomicAdd(&d_biasConvGrad[0], accumBias);

    t += blockDim.x;
  }
}

// aggregateGradientsKernel: no-op (just a sync point between phases)
__global__ void aggregateGradientsKernel(
    float* __restrict__ /*d_fcWeightGrad*/,
    float* __restrict__ /*d_biasFCGrad*/,
    float* __restrict__ /*d_filterGrad*/,
    float* __restrict__ /*d_biasConvGrad*/)
{
  // no-op
}

// updateWeightsKernel: SGD update
__global__ void updateWeightsKernel(
    float* __restrict__ d_filter,
    float* __restrict__ d_biasConv,
    float* __restrict__ d_fcWeight,
    float* __restrict__ d_biasFC,
    const float* __restrict__ d_filterGrad,
    const float* __restrict__ d_biasConvGrad,
    const float* __restrict__ d_fcWeightGrad,
    const float* __restrict__ d_biasFCGrad,
    float learningRate)
{
  // filter
  for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; ++i) {
    d_filter[i] -= learningRate * d_filterGrad[i];
  }
  // conv bias
  d_biasConv[0] -= learningRate * d_biasConvGrad[0];

  // FC weights (NUM_FILTERS=1, FC_OUT=1 demo)
  const int wSize = CONV_OUT_HEIGHT * CONV_OUT_WIDTH;
  for (int w = 0; w < wSize; ++w) {
    d_fcWeight[w] -= learningRate * d_fcWeightGrad[w];
  }
  // FC bias
  d_biasFC[0] -= learningRate * d_biasFCGrad[0];


} // End of int main()

