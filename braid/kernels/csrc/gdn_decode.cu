#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream
#include <c10/cuda/CUDAException.h>  // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <optional>
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

// The ACTIVATION dtype, for reading raw gate projections. Distinct from
// StateIO on purpose: the state pool's dtype and the activation dtype are
// independent axes (an fp16 pool under bf16 activations is the shipping
// configuration), and `round_trip` -- narrow to the activation dtype, widen
// back -- is a gate-specific operation the state never performs. The explicit
// `_rn` conversions are round-to-nearest-even, which is what c10's float->bf16
// and float->half conversions do; the default-rounding intrinsics happen to
// agree today, but the parity claim below should not hang on "happen to".
template <typename T>
struct ActIO;
template <>
struct ActIO<float> {
  __device__ static inline float widen(float v) { return v; }
  __device__ static inline float round_trip(float v) { return v; }
};
template <>
struct ActIO<__nv_bfloat16> {
  __device__ static inline float widen(__nv_bfloat16 v) { return __bfloat162float(v); }
  __device__ static inline float round_trip(float v) {
    return __bfloat162float(__float2bfloat16_rn(v));
  }
};

// --- where alpha and beta come from ------------------------------------------
//
// The recurrence consumes one (alpha, beta) pair per (row, head, token). Where
// that pair comes from is a POLICY template argument, so the recurrence itself
// stays written exactly once and the decode/prefill bit-identity gate keeps
// meaning something:
//
//   PrecomputedGates   loads gates the caller computed in torch. The original
//                      interface; every existing parity test pins it.
//   RawGates<AT>       computes them in-kernel from the raw projections:
//
//                        beta  = widen(round_AT(1 / (1 + exp(-b_raw))))
//                        alpha = exp(A[h] * softplus(a_raw + dt_bias[h]))
//                        softplus(x) = x > 20 ? x : log1p(exp(x))
//
// **Why in-kernel:** the torch spelling of those two lines is ~8 kernel
// launches per GDN layer per step -- sigmoid, two casts, add, softplus, mul,
// exp, contiguous -- ~190 launches across 24 layers, inside a step that is
// already 91% device-busy at B=1. Computed here they are ~4 transcendentals per
// thread against a 2x128-FMA recurrence, and the [B, T, H] alpha/beta tensors
// stop existing at all.
//
// **The numerics are torch's, deliberately.** beta's sigmoid is taken in the
// ACTIVATION dtype and only then widened -- HF's asymmetry, load-bearing
// (braid/model/gdn.py documents the 4.8e-3 shift from doing it in fp32) --
// hence `round_trip`. alpha is fp32 throughout. softplus carries torch's own
// threshold-20 branch. Every primitive maps to the same CUDA math-library call
// torch's own kernels lower to (expf, log1pf, IEEE div; this file builds
// without fast math), so the result is BIT-IDENTICAL to the torch path --
// asserted by tests/test_gdn_raw_gates.py, not argued.
//
// **Padding lives here too.** For a pad column (seq_lens given and
// t >= seq_lens[b]) RawGates yields alpha = 1, beta = 0 -- the exact identity
// step the ragged-prefill contract requires -- replacing the caller's
// torch.where over [B, T, H]. A null seq_lens means every column is live,
// which is what decode (T = 1) always passes.
//
// All HD threads of a block compute the same gate pair redundantly. That is a
// few dozen ALU instructions against the ~700 of the recurrence loop, and the
// alternative -- one thread computes, smem broadcast -- would add a barrier to
// a kernel whose barriers are already load-bearing for the q_hat/k_hat reuse.
struct PrecomputedGates {
  const float* alpha;  // [B, H] at T = 1, [B, T, H] otherwise
  const float* beta;
  __device__ inline void get(int b, int h, int t, int H, int T, float& a, float& bt) const {
    const size_t i = ((size_t)b * T + t) * H + h;
    a = alpha[i];
    bt = beta[i];
  }
};

template <typename AT>
struct RawGates {
  const AT* a_raw;       // [B, T, H], activation dtype
  const AT* b_raw;       // [B, T, H], activation dtype
  const float* A;        // [H], already -exp(A_log)
  const float* dt_bias;  // [H]
  const int* seq_lens;   // [B], or nullptr = all columns live
  __device__ inline void get(int b, int h, int t, int H, int T, float& a, float& bt) const {
    if (seq_lens != nullptr && t >= seq_lens[b]) {
      a = 1.f;  // S' = 1*S + k (x) 0 = S, bit for bit -- the pad identity
      bt = 0.f;
      return;
    }
    const size_t i = ((size_t)b * T + t) * H + h;
    // torch.sigmoid on a reduced-precision tensor computes 1/(1+exp(-x)) in
    // fp32 and rounds the RESULT to the tensor's dtype; the .float() after it
    // widens that rounded value. round_trip reproduces exactly that.
    bt = ActIO<AT>::round_trip(1.f / (1.f + expf(-ActIO<AT>::widen(b_raw[i]))));
    const float x = ActIO<AT>::widen(a_raw[i]) + dt_bias[h];
    // torch's softplus (beta=1, threshold=20), branch included: log1p(exp(x))
    // overflows for large x where the function is x to fp32 precision anyway.
    const float sp = x > 20.f ? x : log1pf(expf(x));
    a = expf(A[h] * sp);
  }
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
// `q_row` / `k_row` / `v_row` are the stride between batch rows, in ELEMENTS.
// Dense inputs pass G*SS / H*HD; the decode path passes the row stride of the
// conv output so that q, k and v can be COLUMN SLICES of the [B, C] conv
// buffer rather than three `.contiguous()` copies of it — 72 copy launches per
// step at B>1 whose only job was to satisfy a contiguity check. The last two
// dims must still be dense; only the batch stride is free.
template <typename ST, typename GS, int SS, int HD>
__global__ void __launch_bounds__(HD, 1) gdn_decode_kernel(
    ST* __restrict__ pool,             // [S_max, H, SS, HD], fp32 / fp16 / bf16
    const int* __restrict__ slot_idx,  // [B]
    const float* __restrict__ q,       // [B, G, SS], row stride q_row
    const float* __restrict__ k,       // [B, G, SS], row stride k_row
    const float* __restrict__ v,       // [B, H, HD], row stride v_row
    GS gates,                          // alpha/beta, loaded or computed -- see above
    float* __restrict__ y,             // [B, H, HD], dense
    int H, int G,
    long q_row, long k_row, long v_row) {
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
  const float* q_in = q + (size_t)b * q_row + (size_t)g * SS;
  const float* k_in = k + (size_t)b * k_row + (size_t)g * SS;

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

  float a, bt;
  gates.get(b, h, /*t=*/0, H, /*T=*/1, a, bt);

  // kv on the UNDECAYED state, then scale by alpha. Order matters for fp32
  // parity: decaying first and then reducing is algebraically identical and
  // not bit-identical.
  float kv = 0.f;
#pragma unroll
  for (int s = 0; s < SS; ++s) kv += S_reg[s] * k_hat[s];
  const float delta = (v[(size_t)b * v_row + (size_t)h * HD + d] - a * kv) * bt;

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

// A [B, G, SS]-shaped operand whose batch rows may live anywhere (a column
// slice of the conv output has row stride C, not G*SS) but whose inner two
// dims must be dense. Returns the row stride in elements. A dense tensor
// passes trivially with stride(0) == inner.
static int64_t row_stride_3d(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32, name, " must be cuda fp32");
  TORCH_CHECK(t.dim() == 3, name, " must be 3-d");
  TORCH_CHECK(t.stride(2) == 1 && t.stride(1) == t.size(2),
              name, " must be dense in its last two dims (only the batch "
              "stride may differ; got strides ", t.strides(), ")");
  TORCH_CHECK(t.stride(0) >= t.size(1) * t.size(2), name, " rows overlap");
  return t.stride(0);
}

void gdn_decode(torch::Tensor pool, torch::Tensor slot_idx, torch::Tensor q, torch::Tensor k,
                torch::Tensor v, torch::Tensor alpha, torch::Tensor beta, torch::Tensor y) {
  TORCH_CHECK(pool.is_cuda(), "pool must be cuda");
  const auto st = pool.scalar_type();
  TORCH_CHECK(st == torch::kFloat32 || st == torch::kFloat16 || st == torch::kBFloat16,
              "state pool must be fp32, fp16 or bf16");
  TORCH_CHECK(slot_idx.scalar_type() == torch::kInt32, "slot_idx must be int32");
  TORCH_CHECK(pool.is_contiguous(), "pool must be contiguous");
  for (const auto& t : {alpha, beta, y}) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(),
                "alpha/beta/y must be contiguous cuda fp32");
  }
  const int64_t q_row = row_stride_3d(q, "q");
  const int64_t k_row = row_stride_3d(k, "k");
  const int64_t v_row = row_stride_3d(v, "v");

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
  const PrecomputedGates gates{alpha.data_ptr<float>(), beta.data_ptr<float>()};

#define BRAID_LAUNCH_GDN(ST, PTR)                                                              \
  gdn_decode_kernel<ST, PrecomputedGates, 128, 128><<<grid, HD, smem, stream>>>(               \
      PTR, slot_idx.data_ptr<int>(), q.data_ptr<float>(), k.data_ptr<float>(),                 \
      v.data_ptr<float>(), gates, y.data_ptr<float>(), H, G, q_row, k_row, v_row)

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

// The raw-gates decode entry: identical recurrence, gates computed in-kernel.
// `a_raw` / `b_raw` are the [B, H] projection outputs in the ACTIVATION dtype
// (bf16 in the shipping engine, fp32 in the fp32 parity engines) -- the bf16
// bits themselves, because beta's sigmoid must round through that dtype to
// match HF. `A` and `dt_bias` are the [H] fp32 layer constants.
void gdn_decode_raw(torch::Tensor pool, torch::Tensor slot_idx, torch::Tensor q,
                    torch::Tensor k, torch::Tensor v, torch::Tensor a_raw,
                    torch::Tensor b_raw, torch::Tensor A, torch::Tensor dt_bias,
                    torch::Tensor y) {
  TORCH_CHECK(pool.is_cuda(), "pool must be cuda");
  const auto st = pool.scalar_type();
  TORCH_CHECK(st == torch::kFloat32 || st == torch::kFloat16 || st == torch::kBFloat16,
              "state pool must be fp32, fp16 or bf16");
  TORCH_CHECK(slot_idx.scalar_type() == torch::kInt32, "slot_idx must be int32");
  TORCH_CHECK(pool.is_contiguous(), "pool must be contiguous");
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32 && y.is_contiguous(),
              "y must be contiguous cuda fp32");
  const int64_t q_row = row_stride_3d(q, "q");
  const int64_t k_row = row_stride_3d(k, "k");
  const int64_t v_row = row_stride_3d(v, "v");
  const auto at = a_raw.scalar_type();
  TORCH_CHECK(at == torch::kFloat32 || at == torch::kBFloat16,
              "a_raw/b_raw must be fp32 or bf16 (the activation dtypes braid runs)");
  TORCH_CHECK(b_raw.scalar_type() == at, "a_raw and b_raw must share a dtype");
  for (const auto& t : {a_raw, b_raw}) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "a_raw/b_raw must be contiguous cuda");
  }
  for (const auto& t : {A, dt_bias}) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(),
                "A/dt_bias must be contiguous cuda fp32");
  }

  const int B = q.size(0), G = q.size(1), SS = q.size(2);
  const int H = v.size(1), HD = v.size(2);
  TORCH_CHECK(SS == 128 && HD == 128, "only SS=HD=128 is instantiated");
  TORCH_CHECK(H % G == 0, "n_heads must be divisible by n_groups");
  TORCH_CHECK(slot_idx.numel() == B, "slot_idx must have one entry per batch row");
  TORCH_CHECK(a_raw.size(0) == B && a_raw.numel() == (int64_t)B * H,
              "a_raw must be [B, H]");
  TORCH_CHECK(b_raw.sizes() == a_raw.sizes(), "b_raw must match a_raw");
  TORCH_CHECK(A.numel() == H && dt_bias.numel() == H, "A and dt_bias must be [H]");
  TORCH_CHECK(pool.size(1) == H && pool.size(2) == SS && pool.size(3) == HD,
              "pool shape must be [S_max, H, SS, HD]");
  TORCH_CHECK(y.size(0) == B && y.size(1) == H && y.size(2) == HD, "y shape must be [B, H, HD]");

  auto stream = at::cuda::getCurrentCUDAStream();
  validate_slots(slot_idx, pool.size(0), stream);

  dim3 grid(B, H);
  const size_t smem = 2 * SS * sizeof(float);

#define BRAID_LAUNCH_GDN_RAW(ST, PTR, AT, APTR, BPTR)                                          \
  gdn_decode_kernel<ST, RawGates<AT>, 128, 128><<<grid, HD, smem, stream>>>(                   \
      PTR, slot_idx.data_ptr<int>(), q.data_ptr<float>(), k.data_ptr<float>(),                 \
      v.data_ptr<float>(),                                                                     \
      RawGates<AT>{APTR, BPTR, A.data_ptr<float>(), dt_bias.data_ptr<float>(),                 \
                   /*seq_lens=*/nullptr},                                                      \
      y.data_ptr<float>(), H, G, q_row, k_row, v_row)

#define BRAID_GDN_RAW_ACT(ST, PTR)                                                             \
  if (at == torch::kFloat32) {                                                                 \
    BRAID_LAUNCH_GDN_RAW(ST, PTR, float, a_raw.data_ptr<float>(), b_raw.data_ptr<float>());    \
  } else {                                                                                     \
    BRAID_LAUNCH_GDN_RAW(ST, PTR, __nv_bfloat16,                                               \
                         reinterpret_cast<const __nv_bfloat16*>(a_raw.data_ptr<at::BFloat16>()),\
                         reinterpret_cast<const __nv_bfloat16*>(b_raw.data_ptr<at::BFloat16>()));\
  }

  if (st == torch::kFloat32) {
    BRAID_GDN_RAW_ACT(float, pool.data_ptr<float>())
  } else if (st == torch::kFloat16) {
    BRAID_GDN_RAW_ACT(__half, reinterpret_cast<__half*>(pool.data_ptr<at::Half>()))
  } else {
    BRAID_GDN_RAW_ACT(__nv_bfloat16,
                      reinterpret_cast<__nv_bfloat16*>(pool.data_ptr<at::BFloat16>()))
  }
#undef BRAID_GDN_RAW_ACT
#undef BRAID_LAUNCH_GDN_RAW
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
template <typename ST, typename GS, int SS, int HD>
__global__ void __launch_bounds__(HD, 1) gdn_prefill_kernel(
    ST* __restrict__ pool,             // [S_max, H, SS, HD]
    const int* __restrict__ slot_idx,  // [B]
    const float* __restrict__ q,       // [B, T, G, SS]
    const float* __restrict__ k,       // [B, T, G, SS]
    const float* __restrict__ v,       // [B, T, H, HD]
    GS gates,                          // PrecomputedGates: pad columns already 1/0.
                                       // RawGates: pad handled via seq_lens.
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
    float a, bt;
    gates.get(b, h, t, H, T, a, bt);

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
  const PrecomputedGates gates{alpha.data_ptr<float>(), beta.data_ptr<float>()};

#define BRAID_LAUNCH_PREFILL(ST, PTR)                                                          \
  gdn_prefill_kernel<ST, PrecomputedGates, 128, 128><<<grid, HD, smem, stream>>>(              \
      PTR, slot_idx.data_ptr<int>(), q.data_ptr<float>(), k.data_ptr<float>(),                 \
      v.data_ptr<float>(), gates, y.data_ptr<float>(), H, G, T)

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

// The raw-gates prefill entry. `seq_lens` (int32 [B], or None) replaces the
// caller's torch.where mask: a pad column becomes alpha=1, beta=0 inside the
// kernel, from the same comparison the mask would have made. `a_raw` / `b_raw`
// are [B, T, H] in the activation dtype, straight off the projection -- the
// sigmoid/softplus/exp chain and the two [B, T, H] fp32 gate tensors it used
// to materialise stop existing.
void gdn_prefill_raw(torch::Tensor pool, torch::Tensor slot_idx, torch::Tensor q,
                     torch::Tensor k, torch::Tensor v, torch::Tensor a_raw,
                     torch::Tensor b_raw, torch::Tensor A, torch::Tensor dt_bias,
                     torch::Tensor y, std::optional<torch::Tensor> seq_lens) {
  TORCH_CHECK(pool.is_cuda(), "pool must be cuda");
  const auto st = pool.scalar_type();
  TORCH_CHECK(st == torch::kFloat32 || st == torch::kFloat16 || st == torch::kBFloat16,
              "state pool must be fp32, fp16 or bf16");
  TORCH_CHECK(slot_idx.scalar_type() == torch::kInt32, "slot_idx must be int32");
  TORCH_CHECK(pool.is_contiguous(), "pool must be contiguous");
  for (const auto& t : {q, k, v, y}) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(),
                "q/k/v/y must be contiguous cuda fp32");
  }
  const auto at = a_raw.scalar_type();
  TORCH_CHECK(at == torch::kFloat32 || at == torch::kBFloat16,
              "a_raw/b_raw must be fp32 or bf16 (the activation dtypes braid runs)");
  TORCH_CHECK(b_raw.scalar_type() == at, "a_raw and b_raw must share a dtype");
  for (const auto& t : {a_raw, b_raw}) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "a_raw/b_raw must be contiguous cuda");
  }
  for (const auto& t : {A, dt_bias}) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(),
                "A/dt_bias must be contiguous cuda fp32");
  }

  const int B = q.size(0), T = q.size(1), G = q.size(2), SS = q.size(3);
  const int H = v.size(2), HD = v.size(3);
  TORCH_CHECK(SS == 128 && HD == 128, "only SS=HD=128 is instantiated");
  TORCH_CHECK(H % G == 0, "n_heads must be divisible by n_groups");
  TORCH_CHECK(slot_idx.numel() == B, "slot_idx must have one entry per batch row");
  TORCH_CHECK(k.sizes() == q.sizes(), "k must match q");
  TORCH_CHECK(v.size(0) == B && v.size(1) == T, "v must match q in [B, T]");
  TORCH_CHECK(a_raw.size(0) == B && a_raw.size(1) == T &&
                  a_raw.numel() == (int64_t)B * T * H,
              "a_raw must be [B, T, H]");
  TORCH_CHECK(b_raw.sizes() == a_raw.sizes(), "b_raw must match a_raw");
  TORCH_CHECK(A.numel() == H && dt_bias.numel() == H, "A and dt_bias must be [H]");
  TORCH_CHECK(pool.size(1) == H && pool.size(2) == SS && pool.size(3) == HD,
              "pool shape must be [S_max, H, SS, HD]");
  TORCH_CHECK(y.size(0) == B && y.size(1) == T && y.size(2) == H && y.size(3) == HD,
              "y shape must be [B, T, H, HD]");

  const int* sl = nullptr;
  if (seq_lens.has_value()) {
    const auto& s = seq_lens.value();
    TORCH_CHECK(s.is_cuda() && s.scalar_type() == torch::kInt32 && s.is_contiguous(),
                "seq_lens must be contiguous cuda int32");
    TORCH_CHECK(s.numel() == B, "seq_lens must have one entry per batch row");
    sl = s.data_ptr<int>();
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  validate_slots(slot_idx, pool.size(0), stream);

  dim3 grid(B, H);
  const size_t smem = 2 * SS * sizeof(float);

#define BRAID_LAUNCH_PREFILL_RAW(ST, PTR, AT, APTR, BPTR)                                      \
  gdn_prefill_kernel<ST, RawGates<AT>, 128, 128><<<grid, HD, smem, stream>>>(                  \
      PTR, slot_idx.data_ptr<int>(), q.data_ptr<float>(), k.data_ptr<float>(),                 \
      v.data_ptr<float>(),                                                                     \
      RawGates<AT>{APTR, BPTR, A.data_ptr<float>(), dt_bias.data_ptr<float>(), sl},            \
      y.data_ptr<float>(), H, G, T)

#define BRAID_PREFILL_RAW_ACT(ST, PTR)                                                         \
  if (at == torch::kFloat32) {                                                                 \
    BRAID_LAUNCH_PREFILL_RAW(ST, PTR, float, a_raw.data_ptr<float>(),                          \
                             b_raw.data_ptr<float>());                                         \
  } else {                                                                                     \
    BRAID_LAUNCH_PREFILL_RAW(ST, PTR, __nv_bfloat16,                                           \
                             reinterpret_cast<const __nv_bfloat16*>(                           \
                                 a_raw.data_ptr<at::BFloat16>()),                              \
                             reinterpret_cast<const __nv_bfloat16*>(                           \
                                 b_raw.data_ptr<at::BFloat16>()));                             \
  }

  if (st == torch::kFloat32) {
    BRAID_PREFILL_RAW_ACT(float, pool.data_ptr<float>())
  } else if (st == torch::kFloat16) {
    BRAID_PREFILL_RAW_ACT(__half, reinterpret_cast<__half*>(pool.data_ptr<at::Half>()))
  } else {
    BRAID_PREFILL_RAW_ACT(__nv_bfloat16,
                          reinterpret_cast<__nv_bfloat16*>(pool.data_ptr<at::BFloat16>()))
  }
#undef BRAID_PREFILL_RAW_ACT
#undef BRAID_LAUNCH_PREFILL_RAW
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
