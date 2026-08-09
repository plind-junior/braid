// Dynamic per-tensor activation quantization, in two launches instead of nine.
//
// **Why this kernel exists, in one measured number.** `scripts/step_profile.py`
// on Qwen3.5-9B at B=1 puts the decode step at 91% device-busy -- so the step is
// not gap-bound and fusing launches for its own sake buys nothing -- but 24.7%
// of that busy time is *elementwise* work over 1,873 launches, against 63.7% in
// the GEMMs. The single largest contributor is this function. The torch spelling
//
//     scale = (x.float().abs().amax() / 448).clamp(min=1e-12).reshape(1, 1)
//     q     = (x.float() / scale).clamp(-448, 448).to(float8_e4m3fn)
//
// is nine kernels -- cast, abs, reduce, div, clamp, reshape, cast, div, clamp,
// convert -- and each of the elementwise ones reads and writes the whole
// activation. It runs ~129 times per decode step, which measured ~1.18 ms of a
// 9.75 ms step. This does it in two passes over the data and two launches.
//
// **The arithmetic is deliberately identical to the torch spelling**, operation
// for operation, because `tests/test_quant.py` gates it on bit-identity against
// that spelling and a per-tensor scale lands straight in every fp8 GEMM:
//   * `fmaxf(|x|)` -- max is associative AND exact in floating point, so the
//     reduction order does not matter and the scale is bit-identical whatever
//     grid this launches with. That is what makes a bitwise gate reasonable here
//     where it would not be for a sum.
//   * The two divisions in the spelling are NOT the same operation in torch,
//     and the kernel mirrors each one. `amax / 448.0` divides a tensor by a
//     CPU scalar, and torch's div kernel computes that as `amax * (1/448)` --
//     one fp32 reciprocal, then a multiply -- which differs from true IEEE
//     division by up to 1 ulp. `x / scale` divides by a CUDA *tensor*, which
//     IS true elementwise division. First implementation used true division
//     for both; the bit-identity gate failed with the scale exactly 1 ulp low
//     on ~60% of inputs, which is precisely the reciprocal-multiply signature.
//     Reproducing torch's approximation is the contract; improving on it is a
//     silent numerics change.
//   * clamp-then-convert, not convert-then-saturate, even though `__NV_SATFINITE`
//     makes the clamp redundant for finite inputs. Matching the spelling is
//     worth more than one instruction.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

namespace {

constexpr float kFp8Max = 448.0f;
// torch's tensor-by-cpu-scalar division is `t * (1/scalar)` at opmath
// precision, not true division. This constant is that reciprocal, rounded
// once at compile time exactly as torch's div kernel rounds it at runtime.
constexpr float kInvFp8Max = 1.0f / 448.0f;
constexpr float kEps = 1e-12f;
constexpr int kThreads = 256;
// Bounds the partial buffer so pass 2 can re-reduce it cheaply from every block.
// 256 floats is 1 KiB and sits in L2; the alternative -- a third launch to
// finalize the scale -- would give back a third of what this kernel is for.
constexpr int kMaxBlocks = 256;

__device__ __forceinline__ float widen(float v) { return v; }
__device__ __forceinline__ float widen(__half v) { return __half2float(v); }
__device__ __forceinline__ float widen(__nv_bfloat16 v) { return __bfloat162float(v); }

// **NaN-propagating max, deliberately not `fmaxf`.** `fmaxf(NaN, x)` returns x,
// so a reduction built on it *swallows* NaN — while `torch.amax` returns NaN.
// Matching torch is not pedantry here: the scale divides every element, so a
// swallowed NaN turns a diverged activation into a full tensor of plausible
// -448..448 fp8 and the run keeps going looking healthy. A kernel that hides
// the evidence of divergence is the exact silent-wrong-output failure this
// project's kernel notes warn about. One extra comparison in a memory-bound
// loop is not a price worth arguing over.
//
// `v != v` is the NaN test. Once `m` is NaN it stays NaN: for finite `v`,
// `v > NaN` is false and `v != v` is false, so neither branch replaces it.
__device__ __forceinline__ float nan_max(float m, float v) {
  return (v != v || v > m) ? v : m;
}

__device__ __forceinline__ float block_max(float m) {
  __shared__ float s[kThreads];
  const int t = threadIdx.x;
  s[t] = m;
  __syncthreads();
#pragma unroll
  for (int off = kThreads / 2; off > 0; off >>= 1) {
    if (t < off) s[t] = nan_max(s[t], s[t + off]);
    __syncthreads();
  }
  return s[0];
}

template <typename T>
__global__ void act_amax_partial(const T* __restrict__ x, float* __restrict__ partial,
                                 long n) {
  float m = 0.f;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (long)gridDim.x * blockDim.x) {
    m = nan_max(m, fabsf(widen(x[i])));
  }
  m = block_max(m);
  if (threadIdx.x == 0) partial[blockIdx.x] = m;
}

template <typename T>
__global__ void act_quant(const T* __restrict__ x, const float* __restrict__ partial,
                          int nparts, float* __restrict__ scale,
                          __nv_fp8_storage_t* __restrict__ out, long n) {
  // Every block re-reduces the partials. Redundant by design: it removes a
  // launch, the buffer is at most 1 KiB, and the result is bit-identical in
  // every block because max is exact.
  float m = 0.f;
  for (int i = threadIdx.x; i < nparts; i += kThreads) m = nan_max(m, partial[i]);
  m = block_max(m);

  // `.clamp(min=EPS)` in torch leaves NaN alone; `fmaxf` would replace it with
  // EPS. Written as a comparison so a NaN amax stays NaN and the divergence
  // reaches the output instead of being rounded away here.
  //
  // Multiply, not divide -- torch's own semantics for scalar division; see the
  // header. (NaN * kInvFp8Max is still NaN, so the propagation argument holds.)
  const float raw = m * kInvFp8Max;
  const float sc = (raw < kEps) ? kEps : raw;
  if (blockIdx.x == 0 && threadIdx.x == 0) scale[0] = sc;

  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (long)gridDim.x * blockDim.x) {
    float v = widen(x[i]) / sc;
    // Same reason: `torch.clamp` propagates NaN, `fminf`/`fmaxf` do not. Both
    // comparisons are false for NaN, so it falls through untouched.
    v = (v < -kFp8Max) ? -kFp8Max : ((v > kFp8Max) ? kFp8Max : v);
    out[i] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  }
}

int grid_for(long n) {
  const long want = (n + kThreads - 1) / kThreads;
  return (int)(want < 1 ? 1 : (want > kMaxBlocks ? kMaxBlocks : want));
}

}  // namespace

std::vector<torch::Tensor> quantize_act_cuda(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be cuda");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  const auto st = x.scalar_type();
  TORCH_CHECK(st == torch::kFloat32 || st == torch::kBFloat16 || st == torch::kFloat16,
              "activation must be fp32, bf16 or fp16");

  const long n = x.numel();
  TORCH_CHECK(n > 0, "x must not be empty");

  auto opts = x.options();
  auto out = torch::empty(x.sizes(), opts.dtype(torch::kFloat8_e4m3fn));
  auto scale = torch::empty({1, 1}, opts.dtype(torch::kFloat32));
  const int blocks = grid_for(n);
  auto partial = torch::empty({blocks}, opts.dtype(torch::kFloat32));

  auto stream = at::cuda::getCurrentCUDAStream();
  auto* op = reinterpret_cast<__nv_fp8_storage_t*>(out.data_ptr());

#define BRAID_LAUNCH_QUANT(T, PTR)                                                    \
  act_amax_partial<T><<<blocks, kThreads, 0, stream>>>(PTR, partial.data_ptr<float>(), \
                                                       n);                             \
  act_quant<T><<<blocks, kThreads, 0, stream>>>(                                       \
      PTR, partial.data_ptr<float>(), blocks, scale.data_ptr<float>(), op, n)

  if (st == torch::kFloat32) {
    BRAID_LAUNCH_QUANT(float, x.data_ptr<float>());
  } else if (st == torch::kFloat16) {
    BRAID_LAUNCH_QUANT(__half, reinterpret_cast<const __half*>(x.data_ptr<at::Half>()));
  } else {
    BRAID_LAUNCH_QUANT(__nv_bfloat16,
                       reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
  }
#undef BRAID_LAUNCH_QUANT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, scale};
}
