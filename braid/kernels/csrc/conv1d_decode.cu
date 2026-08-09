#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Batched slotted causal conv1d decode step, SiLU fused, fp32 out.
//
// Mirrors the reference engine's ssm_conv1d_decode_f32_silu (ssm.cu:145-181):
// one thread per channel, slide the per-channel window left by one, append the
// new token's
// value, dot against the weight row, add bias, apply SiLU. Two differences:
// a batch axis on the grid, and the state slot read from device memory so the
// kernel is CUDA-graph-stable across slot reassignment.
//
// Their decode conv is not batchable as written for a subtler reason than the
// scan: it derives `channels` from x.shape[ndim-1] and launches
// ceil(channels/256) blocks with one thread per channel, so a [B, C] input
// silently processes only row 0. Not a crash -- just row 0's answer, B times.
//
// Layout notes that a reimplementation gets wrong silently:
//   conv_state[ch * K + k]  -- K contiguous per channel, matching HF's
//                              [C, 1, K] flattened. A transposed read is
//                              plausible and wrong.
//   bias is added AFTER the dot and BEFORE the SiLU.
//   SiLU is applied to ALL channels -- Q, K and V alike -- matching
//   causal_conv1d_fn(activation='silu'), not just to the V block.

constexpr int kMaxConvKernel = 8;

// The input may arrive in the ACTIVATION dtype (bf16 in the shipping engine)
// and is widened per element here. This is exactly the `.float()` the caller
// used to run as a separate kernel over the whole [B, C] tensor before every
// conv — 24 cast launches per decode step whose only job was to feed this
// kernel. bf16 -> fp32 widening is exact, so reading bf16 here is bit-identical
// to casting first; the cast kernel simply stops existing.
__device__ __forceinline__ float conv_widen(float v) { return v; }
__device__ __forceinline__ float conv_widen(__nv_bfloat16 v) { return __bfloat162float(v); }

template <int K, typename XT>
__global__ void conv1d_decode_kernel(
    float* __restrict__ conv_pool,     // [S_max, C, K]
    const int* __restrict__ slot_idx,  // [B]
    const XT* __restrict__ x,          // [B, C], fp32 or bf16
    const float* __restrict__ weight,  // [C, K]
    const float* __restrict__ bias,    // [C] or nullptr
    float* __restrict__ out,           // [B, C]
    int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int b = blockIdx.y;

  const int slot = slot_idx[b];
  float* st = conv_pool + ((size_t)slot * C + c) * K;

  // Slide in registers, then write back once. Doing the shift in place in
  // global memory (as the reference engine does) is only safe because each
  // thread's read index stays ahead of its write index; staging through
  // registers removes the
  // constraint and costs nothing at K=4.
  float w[K];
#pragma unroll
  for (int i = 0; i < K - 1; ++i) w[i] = st[i + 1];
  w[K - 1] = conv_widen(x[(size_t)b * C + c]);

  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < K; ++i) {
    st[i] = w[i];
    acc += w[i] * weight[(size_t)c * K + i];
  }
  if (bias != nullptr) acc += bias[c];

  out[(size_t)b * C + c] = acc / (1.f + __expf(-acc));  // SiLU
}

static void validate_slots_conv(const torch::Tensor& slot_idx, int64_t n_slots,
                                cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
  if (status != cudaStreamCaptureStatusNone) return;

  const torch::Tensor host = slot_idx.to(torch::kCPU);
  const int* p = host.data_ptr<int>();
  for (int64_t i = 0; i < host.numel(); ++i) {
    TORCH_CHECK(p[i] >= 0 && p[i] < n_slots, "slot_idx[", i, "] = ", p[i], " is out of range [0, ",
                n_slots, ")");
  }
}

void conv1d_decode(torch::Tensor conv_pool, torch::Tensor slot_idx, torch::Tensor x,
                   torch::Tensor weight, torch::Tensor bias, torch::Tensor out) {
  TORCH_CHECK(conv_pool.is_cuda() && conv_pool.scalar_type() == torch::kFloat32,
              "conv_pool must be cuda fp32");
  TORCH_CHECK(conv_pool.is_contiguous(), "conv_pool must be contiguous");
  TORCH_CHECK(slot_idx.scalar_type() == torch::kInt32, "slot_idx must be int32");
  TORCH_CHECK(conv_pool.dim() == 3, "conv_pool must be [S_max, C, K]");

  const int64_t n_slots = conv_pool.size(0);
  const int C = conv_pool.size(1);
  const int K = conv_pool.size(2);
  const int B = x.size(0);

  TORCH_CHECK(K <= kMaxConvKernel, "conv_kernel > ", kMaxConvKernel, " is not instantiated");
  TORCH_CHECK(x.size(1) == C && out.size(0) == B && out.size(1) == C, "x/out must be [B, C]");
  TORCH_CHECK(weight.size(0) == C && weight.size(1) == K, "weight must be [C, K]");
  TORCH_CHECK(slot_idx.numel() == B, "slot_idx must have one entry per batch row");
  const auto xt = x.scalar_type();
  TORCH_CHECK(xt == torch::kFloat32 || xt == torch::kBFloat16,
              "x must be fp32 or bf16 (the activation dtypes braid runs)");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32 && out.is_contiguous(),
              "out must be contiguous fp32");

  auto stream = at::cuda::getCurrentCUDAStream();
  validate_slots_conv(slot_idx, n_slots, stream);

  const int threads = 256;
  dim3 grid((C + threads - 1) / threads, B);
  const float* bias_p = bias.defined() && bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;

#define LAUNCH(KK, XT, XPTR)                                                                    \
  conv1d_decode_kernel<KK, XT><<<grid, threads, 0, stream>>>(                                   \
      conv_pool.data_ptr<float>(), slot_idx.data_ptr<int>(), XPTR,                              \
      weight.data_ptr<float>(), bias_p, out.data_ptr<float>(), C);                              \
  break;

  if (xt == torch::kFloat32) {
    switch (K) {
      case 2: LAUNCH(2, float, x.data_ptr<float>())
      case 3: LAUNCH(3, float, x.data_ptr<float>())
      case 4: LAUNCH(4, float, x.data_ptr<float>())
      case 8: LAUNCH(8, float, x.data_ptr<float>())
      default: TORCH_CHECK(false, "conv_kernel ", K, " is not instantiated (have 2, 3, 4, 8)");
    }
  } else {
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    switch (K) {
      case 2: LAUNCH(2, __nv_bfloat16, xp)
      case 3: LAUNCH(3, __nv_bfloat16, xp)
      case 4: LAUNCH(4, __nv_bfloat16, xp)
      case 8: LAUNCH(8, __nv_bfloat16, xp)
      default: TORCH_CHECK(false, "conv_kernel ", K, " is not instantiated (have 2, 3, 4, 8)");
    }
  }
#undef LAUNCH

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
