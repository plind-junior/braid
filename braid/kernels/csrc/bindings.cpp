#include <torch/extension.h>

#include <optional>

void gdn_decode(torch::Tensor pool, torch::Tensor slot_idx, torch::Tensor q, torch::Tensor k,
                torch::Tensor v, torch::Tensor alpha, torch::Tensor beta, torch::Tensor y);

void gdn_prefill(torch::Tensor pool, torch::Tensor slot_idx, torch::Tensor q, torch::Tensor k,
                 torch::Tensor v, torch::Tensor alpha, torch::Tensor beta, torch::Tensor y);

void gdn_decode_raw(torch::Tensor pool, torch::Tensor slot_idx, torch::Tensor q, torch::Tensor k,
                    torch::Tensor v, torch::Tensor a_raw, torch::Tensor b_raw, torch::Tensor A,
                    torch::Tensor dt_bias, torch::Tensor y);

void gdn_prefill_raw(torch::Tensor pool, torch::Tensor slot_idx, torch::Tensor q, torch::Tensor k,
                     torch::Tensor v, torch::Tensor a_raw, torch::Tensor b_raw, torch::Tensor A,
                     torch::Tensor dt_bias, torch::Tensor y,
                     std::optional<torch::Tensor> seq_lens);

void conv1d_decode(torch::Tensor conv_pool, torch::Tensor slot_idx, torch::Tensor x,
                   torch::Tensor weight, torch::Tensor bias, torch::Tensor out);

std::vector<torch::Tensor> quantize_act_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_decode", &gdn_decode, "batched GDN decode step");
  m.def("gdn_prefill", &gdn_prefill, "batched GDN chunk scan, T tokens per launch");
  m.def("gdn_decode_raw", &gdn_decode_raw,
        "GDN decode step with alpha/beta computed in-kernel from the raw projections");
  m.def("gdn_prefill_raw", &gdn_prefill_raw,
        "GDN chunk scan with in-kernel gates and seq_lens padding");
  m.def("conv1d_decode", &conv1d_decode, "batched slotted causal conv1d decode step + SiLU");
  m.def("quantize_act", &quantize_act_cuda,
        "dynamic per-tensor fp8 activation quantization, two launches");
}
