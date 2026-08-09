#include <torch/extension.h>

void gdn_decode(torch::Tensor pool, torch::Tensor slot_idx, torch::Tensor q, torch::Tensor k,
                torch::Tensor v, torch::Tensor alpha, torch::Tensor beta, torch::Tensor y);

void gdn_prefill(torch::Tensor pool, torch::Tensor slot_idx, torch::Tensor q, torch::Tensor k,
                 torch::Tensor v, torch::Tensor alpha, torch::Tensor beta, torch::Tensor y);

void conv1d_decode(torch::Tensor conv_pool, torch::Tensor slot_idx, torch::Tensor x,
                   torch::Tensor weight, torch::Tensor bias, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_decode", &gdn_decode, "batched GDN decode step");
  m.def("gdn_prefill", &gdn_prefill, "batched GDN chunk scan, T tokens per launch");
  m.def("conv1d_decode", &conv1d_decode, "batched slotted causal conv1d decode step + SiLU");
}
