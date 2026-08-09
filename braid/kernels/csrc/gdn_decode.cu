#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream
#include <c10/cuda/CUDAException.h>  // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

// The state pool's STORAGE type. The recurrence itself is fp32 whatever this
// is: the column is widened into registers on load and narrowed once on store,
// so a 16-bit pool costs one rounding per step rather than a lower-precision
// scan. It exists because the state is the largest per-sequence term -- 48 MiB
// per sequence on Qwen3.5-9B, read and written every step, which at B=64 is
// 6.14 GiB against 7.40 GiB of fp8 weights.
template <typename T>
struct StateIO {
  __device__ static inline float load(const T* p) { return static_cast<float>(*p); }
  __device__ static inline void store(T* p, float v) { *p = static_cast<T>(v); }
};
template <>
struct StateIO<float> {
  __device__ static inline float load(const float* p) { return *p; }
  __device__ static inline void store(float* p, float v) { *p = v; }
};
template <>
struct StateIO<__half> {
  __device__ static inline float load(const __half* p) { return __half2float(*p); }
  __device__ static inline void store(__half* p, float v) { *p = __float2half(v); }
};
template <>
struct StateIO<__nv_bfloat16> {
  __device__ static inline float load(const __nv_bfloat16* p) { return __bfloat162float(*p); }
  __device__ static inline void store(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
};

// One block per (batch row, head). One thread per head_dim column d.
// Each thread holds the state column S[:, d] (state_size floats) in registers.
// Mirrors the reference engine's gdn_scan_fused_kernel ownership model at
// n_tokens == 1, but with a batch axis on the grid and the state slot read from
// device memory.
//
// __launch_bounds__(HD, 1) is MANDATORY and the second argument must be 1. The
// reference engine records that __launch_bounds__(HD, 2) at HD=128 is a ptxas
// MISCOMPILE on sm_120 -- the kernel math is correct and the output is garbage
// (.claude/skills/sm120-cuda-expert/references/known-issues.md:55). The min-1
// form is also what lets ptxas give each thread the ~128 registers S_reg needs
// without spilling it to local memory.
template <typename ST, int SS, int HD>
__global__ void __launch_bounds__(HD, 1) gdn_decode_kernel(
    ST* __restrict__ pool,             // [S_max, H, SS, HD], fp32 / fp16 / bf16
    const int* __restrict__ slot_idx,  // [B]
    const float* __restrict__ q,       // [B, G, SS]
    const float* __restrict__ k,       // [B, G, SS]
    const float* __restrict__ v,       // [B, H, HD]
    const float* __restrict__ alpha,   // [B, H]
    const float* __restrict__ beta,    // [B, H]
    float* __restrict__ y,             // [B, H, HD]
    int H, int G) {
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int d = threadIdx.x;
  // GROUPED (HF SafeTensors) layout. The tiled (GGUF) layout is h % G and is a
  // different permutation of the same index range -- it produces plausible
  // garbage, never a crash (reference engine, gdn.cu:55).
  const int g = h / (H / G);

  extern __shared__ float smem[];
  float* q_hat = smem;       // [SS]
  float* k_hat = smem + SS;  // [SS]

  // Clamped L2 normalize q and k over SS.
  //
  // DETERMINISTIC reduction on purpose. atomicAdd on float is
  // order-nondeterministic, which would make the parity tests intermittently
  // flaky at rtol=2e-5 and look like a kernel bug. A fixed-order tree
  // reduction gives the same bits every run.
  //
  // q and k use SEPARATE accumulator arrays. The reference engine reduces both
  // through one s_reduce[] and races: after the k-reduction's final
  // __syncthreads() every
  // thread reads s_reduce[0] and immediately writes s_reduce[d] with no
  // barrier between (gdn.cu:129-131 and three sibling sites). Two arrays cost
  // 512 extra bytes of shared memory and remove the hazard entirely.
  const float* q_in = q + (size_t)b * G * SS + (size_t)g * SS;
  const float* k_in = k + (size_t)b * G * SS + (size_t)g * SS;

  __shared__ float red_q[HD];
  __shared__ float red_k[HD];
  float qs = 0.f, ks = 0.f;
  for (int s = d; s < SS; s += HD) {
    qs += q_in[s] * q_in[s];
    ks += k_in[s] * k_in[s];
  }
  red_q[d] = qs;
  red_k[d] = ks;
  __syncthreads();
#pragma unroll
  for (int off = HD / 2; off > 0; off >>= 1) {
    if (d < off) {
      red_q[d] += red_q[d + off];
      red_k[d] += red_k[d + off];
    }
    __syncthreads();
  }
  // ADDITIVE epsilon on the sum of squares, matching HF's l2norm exactly:
  //   inv_norm = rsqrt((x*x).sum(-1) + 1e-6)
  // (transformers.models.qwen3_5.modeling_qwen3_5.l2norm)
  //
  // The reference engine uses a CLAMP instead -- rsqrtf(fmaxf(sum_sq, 1e-12f))
  // -- and documents an argument for it (gdn.cu:126-130). The two agree to
  // ~4e-9 on a healthy head (sum_sq ~ 128) and differ by 10x on a near-zero
  // head (sum_sq ~ 1e-8: HF 995 vs 10000). HF is the implementation this
  // checkpoint was trained with, so HF wins and theirs is the deviation.
  const float q_inv = rsqrtf(red_q[0] + 1e-6f);
  const float k_inv = rsqrtf(red_k[0] + 1e-6f);
  for (int s = d; s < SS; s += HD) {
    q_hat[s] = q_in[s] * q_inv;
    k_hat[s] = k_in[s] * k_inv;
  }
  __syncthreads();

  // The whole point: the slot is read from DEVICE memory, not baked into the
  // capture as a kernel parameter. One captured graph therefore stays valid
  // for any assignment of sequences to slots. The reference engine re-captures
  // on every rotation at ~10-20 ms (engine_scheduler.cpp:1968-1981,
  // config.h:130-138).
  const int slot = slot_idx[b];
  ST* H_col = pool + ((size_t)slot * H + h) * SS * HD + d;

  // Widened on load, narrowed once on store. Everything between is fp32.
  float S_reg[SS];
#pragma unroll
  for (int s = 0; s < SS; ++s) S_reg[s] = StateIO<ST>::load(H_col + (size_t)s * HD);

  const float a = alpha[(size_t)b * H + h];
  const float bt = beta[(size_t)b * H + h];

  // kv on the UNDECAYED state, then scale by alpha. Order matters for fp32
  // parity: decaying first and then reducing is algebraically identical and
  // not bit-identical.
  float kv = 0.f;
#pragma unroll
  for (int s = 0; s < SS; ++s) kv += S_reg[s] * k_hat[s];
  const float delta = (v[((size_t)b * H + h) * HD + d] - a * kv) * bt;

  // State update and y accumulation share ONE loop, as the reference engine
  // does (gdn.cu:148-174). Splitting into "update then read" changes the fp32
  // result.
  float acc = 0.f;
#pragma unroll
  for (int s = 0; s < SS; ++s) {
    const float nv = a * S_reg[s] + k_hat[s] * delta;
    S_reg[s] = nv;
    acc += nv * q_hat[s];
  }

#pragma unroll
  for (int s = 0; s < SS; ++s) StateIO<ST>::store(H_col + (size_t)s * HD, S_reg[s]);
  // The 1/sqrt scale uses the VALUE head dim and is applied to y AFTER the
  // q_hat dot product, using the NEW state. HF/fla scale q before the dot;
  // identical when head_k_dim == head_v_dim (both 128 here) and divergent
  // otherwise.
  y[((size_t)b * H + h) * HD + d] = acc * rsqrtf((float)HD);
}

// Slot validation costs a device-to-host copy, which is an ILLEGAL MEMORY
// ACCESS inside a captured region -- and on WSL2 compute-sanitizer cannot even
// diagnose it (reference engine, known-issues.md:64). So it runs only when the
// stream is not capturing: full checking in normal use, zero cost and
// graph-safety under capture. This mirrors the reference engine's capture-guard
// pattern for its dequant workspace.
static void validate_slots(const torch::Tensor& slot_idx, int64_t n_slots, cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
  if (status != cudaStreamCaptureStatusNone) return;

  const torch::Tensor host = slot_idx.to(torch::kCPU);
  const int* p = host.data_ptr<int>();
  const int64_t B = host.numel();
  std::vector<char> seen(n_slots, 0);
  for (int64_t i = 0; i < B; ++i) {
    TORCH_CHECK(p[i] >= 0 && p[i] < n_slots, "slot_idx[", i, "] = ", p[i],
                " is out of range [0, ", n_slots, ")");
    TORCH_CHECK(!seen[p[i]], "slot_idx has duplicate slot ", p[i],
                "; two live rows sharing one state slab is silent "
                "cross-sequence corruption. Slots must be distinct.");
    seen[p[i]] = 1;
  }
}

void gdn_decode(torch::Tensor pool, torch::Tensor slot_idx, torch::Tensor q, torch::Tensor k,
                torch::Tensor v, torch::Tensor alpha, torch::Tensor beta, torch::Tensor y) {
  TORCH_CHECK(pool.is_cuda(), "pool must be cuda");
  const auto st = pool.scalar_type();
  TORCH_CHECK(st == torch::kFloat32 || st == torch::kFloat16 || st == torch::kBFloat16,
              "state pool must be fp32, fp16 or bf16");
  TORCH_CHECK(slot_idx.scalar_type() == torch::kInt32, "slot_idx must be int32");
  TORCH_CHECK(pool.is_contiguous(), "pool must be contiguous");
  for (const auto& t : {q, k, v, alpha, beta, y}) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(),
                "all operands must be contiguous cuda fp32");
  }

  const int B = q.size(0), G = q.size(1), SS = q.size(2);
  const int H = v.size(1), HD = v.size(2);
  TORCH_CHECK(SS == 128 && HD == 128, "only SS=HD=128 is instantiated");
  TORCH_CHECK(H % G == 0, "n_heads must be divisible by n_groups");
  TORCH_CHECK(slot_idx.numel() == B, "slot_idx must have one entry per batch row");
  TORCH_CHECK(pool.size(1) == H && pool.size(2) == SS && pool.size(3) == HD,
              "pool shape must be [S_max, H, SS, HD]");
  TORCH_CHECK(y.size(0) == B && y.size(1) == H && y.size(2) == HD, "y shape must be [B, H, HD]");

  auto stream = at::cuda::getCurrentCUDAStream();
  validate_slots(slot_idx, pool.size(0), stream);

  dim3 grid(B, H);
  const size_t smem = 2 * SS * sizeof(float);

#define BRAID_LAUNCH_GDN(ST, PTR)                                                              \
  gdn_decode_kernel<ST, 128, 128><<<grid, HD, smem, stream>>>(                                 \
      PTR, slot_idx.data_ptr<int>(), q.data_ptr<float>(), k.data_ptr<float>(),                 \
      v.data_ptr<float>(), alpha.data_ptr<float>(), beta.data_ptr<float>(),                    \
      y.data_ptr<float>(), H, G)

  if (st == torch::kFloat32) {
    BRAID_LAUNCH_GDN(float, pool.data_ptr<float>());
  } else if (st == torch::kFloat16) {
    BRAID_LAUNCH_GDN(__half, reinterpret_cast<__half*>(pool.data_ptr<at::Half>()));
  } else {
    BRAID_LAUNCH_GDN(__nv_bfloat16,
                     reinterpret_cast<__nv_bfloat16*>(pool.data_ptr<at::BFloat16>()));
  }
#undef BRAID_LAUNCH_GDN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// PREFILL: the same scalar recurrence, T tokens deep, with the state held in
// registers for the whole chunk.
//
// **The point is the launch count, not the arithmetic.** braid's prefill runs
// the one-token step in a Python loop over the sequence axis: one iteration per
// *column*, each a handful of small kernels. That is what makes prefill
// launch-bound, what made batching the rows worth 6.96x, and what still leaves
// prefill at 70% of the served wall clock at c=64 after that batching. This
// kernel keeps the identical scalar recurrence and moves the loop inside, so a
// [B, T] chunk costs ONE launch per layer instead of T iterations of several.
//
// It is deliberately not a WY / UT-transform / tensor-core chunkwise scan.
// Every tensor-core variant this project has measured loses to a plain
// chunk-cached scalar loop on this card, and a different recurrence would also
// have to be re-proved against the oracle. The arithmetic here is line-for-line
// `gdn_decode_kernel`, so prefill and decode stay the same function by
// construction -- which is the property `tests/test_chunked_prefill.py` exists
// to defend.
//
// Padding is handled by the CALLER, which passes alpha = 1 and beta = 0 for a
// pad column. That makes the step an exact identity on the state (delta = 0,
// S' = S bit-for-bit) rather than an approximate one, and it keeps the ragged
// masking in one place instead of two.
template <typename ST, int SS, int HD>
__global__ void __launch_bounds__(HD, 1) gdn_prefill_kernel(
    ST* __restrict__ pool,             // [S_max, H, SS, HD]
    const int* __restrict__ slot_idx,  // [B]
    const float* __restrict__ q,       // [B, T, G, SS]
    const float* __restrict__ k,       // [B, T, G, SS]
    const float* __restrict__ v,       // [B, T, H, HD]
    const float* __restrict__ alpha,   // [B, T, H], pad columns already 1
    const float* __restrict__ beta,    // [B, T, H], pad columns already 0
    float* __restrict__ y,             // [B, T, H, HD]
    int H, int G, int T) {
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int d = threadIdx.x;
  const int g = h / (H / G);

  extern __shared__ float smem[];
  float* q_hat = smem;       // [SS]
  float* k_hat = smem + SS;  // [SS]
  __shared__ float red_q[HD];
  __shared__ float red_k[HD];

  const int slot = slot_idx[b];
  ST* H_col = pool + ((size_t)slot * H + h) * SS * HD + d;

  // Loaded once for the whole chunk and stored once at the end. This is the
  // saving: the decode kernel pays this round trip per token.
  float S_reg[SS];
#pragma unroll
  for (int s = 0; s < SS; ++s) S_reg[s] = StateIO<ST>::load(H_col + (size_t)s * HD);

  for (int t = 0; t < T; ++t) {
    const float* q_in = q + (((size_t)b * T + t) * G + g) * SS;
    const float* k_in = k + (((size_t)b * T + t) * G + g) * SS;

    float qs = 0.f, ks = 0.f;
    for (int s = d; s < SS; s += HD) {
      qs += q_in[s] * q_in[s];
      ks += k_in[s] * k_in[s];
    }
    red_q[d] = qs;
    red_k[d] = ks;
    __syncthreads();
#pragma unroll
    for (int off = HD / 2; off > 0; off >>= 1) {
      if (d < off) {
        red_q[d] += red_q[d + off];
        red_k[d] += red_k[d + off];
      }
      __syncthreads();
    }
    const float q_inv = rsqrtf(red_q[0] + 1e-6f);
    const float k_inv = rsqrtf(red_k[0] + 1e-6f);
    for (int s = d; s < SS; s += HD) {
      q_hat[s] = q_in[s] * q_inv;
      k_hat[s] = k_in[s] * k_inv;
    }
    __syncthreads();

    const size_t bth = ((size_t)b * T + t) * H + h;
    const float a = alpha[bth];
    const float bt = beta[bth];

    float kv = 0.f;
#pragma unroll
    for (int s = 0; s < SS; ++s) kv += S_reg[s] * k_hat[s];
    const float delta = (v[bth * HD + d] - a * kv) * bt;

    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < SS; ++s) {
      const float nv = a * S_reg[s] + k_hat[s] * delta;
      S_reg[s] = nv;
      acc += nv * q_hat[s];
    }
    y[bth * HD + d] = acc * rsqrtf((float)HD);

    // q_hat / k_hat / red_* are reused by the next token; nothing may race
    // ahead and overwrite them while a slower thread is still reading.
    __syncthreads();
  }

#pragma unroll
  for (int s = 0; s < SS; ++s) StateIO<ST>::store(H_col + (size_t)s * HD, S_reg[s]);
}

void gdn_prefill(torch::Tensor pool, torch::Tensor slot_idx, torch::Tensor q, torch::Tensor k,
                 torch::Tensor v, torch::Tensor alpha, torch::Tensor beta, torch::Tensor y) {
  TORCH_CHECK(pool.is_cuda(), "pool must be cuda");
  const auto st = pool.scalar_type();
  TORCH_CHECK(st == torch::kFloat32 || st == torch::kFloat16 || st == torch::kBFloat16,
              "state pool must be fp32, fp16 or bf16");
  TORCH_CHECK(slot_idx.scalar_type() == torch::kInt32, "slot_idx must be int32");
  TORCH_CHECK(pool.is_contiguous(), "pool must be contiguous");
  for (const auto& t : {q, k, v, alpha, beta, y}) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(),
                "all operands must be contiguous cuda fp32");
  }

  const int B = q.size(0), T = q.size(1), G = q.size(2), SS = q.size(3);
  const int H = v.size(2), HD = v.size(3);
  TORCH_CHECK(SS == 128 && HD == 128, "only SS=HD=128 is instantiated");
  TORCH_CHECK(H % G == 0, "n_heads must be divisible by n_groups");
  TORCH_CHECK(slot_idx.numel() == B, "slot_idx must have one entry per batch row");
  TORCH_CHECK(k.size(0) == B && k.size(1) == T, "k must match q in [B, T]");
  TORCH_CHECK(v.size(0) == B && v.size(1) == T, "v must match q in [B, T]");
  TORCH_CHECK(alpha.size(0) == B && alpha.size(1) == T && alpha.size(2) == H,
              "alpha must be [B, T, H]");
  TORCH_CHECK(beta.sizes() == alpha.sizes(), "beta must match alpha");
  TORCH_CHECK(pool.size(1) == H && pool.size(2) == SS && pool.size(3) == HD,
              "pool shape must be [S_max, H, SS, HD]");
  TORCH_CHECK(y.size(0) == B && y.size(1) == T && y.size(2) == H && y.size(3) == HD,
              "y shape must be [B, T, H, HD]");

  auto stream = at::cuda::getCurrentCUDAStream();
  validate_slots(slot_idx, pool.size(0), stream);

  dim3 grid(B, H);
  const size_t smem = 2 * SS * sizeof(float);

#define BRAID_LAUNCH_PREFILL(ST, PTR)                                                          \
  gdn_prefill_kernel<ST, 128, 128><<<grid, HD, smem, stream>>>(                                \
      PTR, slot_idx.data_ptr<int>(), q.data_ptr<float>(), k.data_ptr<float>(),                 \
      v.data_ptr<float>(), alpha.data_ptr<float>(), beta.data_ptr<float>(),                    \
      y.data_ptr<float>(), H, G, T)

  if (st == torch::kFloat32) {
    BRAID_LAUNCH_PREFILL(float, pool.data_ptr<float>());
  } else if (st == torch::kFloat16) {
    BRAID_LAUNCH_PREFILL(__half, reinterpret_cast<__half*>(pool.data_ptr<at::Half>()));
  } else {
    BRAID_LAUNCH_PREFILL(__nv_bfloat16,
                         reinterpret_cast<__nv_bfloat16*>(pool.data_ptr<at::BFloat16>()));
  }
#undef BRAID_LAUNCH_PREFILL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
