#include <torch/extension.h>
#include <vector>

// std::vector<torch::Tensor> slstm_cuda_forward(
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor bias,
//     torch::Tensor old_h,
//     torch::Tensor old_cell);

// std::vector<torch::Tensor> slstm_cuda_backward(
//     at::Tensor grad_h,
//     at::Tensor grad_cell,
//     at::Tensor cell_t,
//     at::Tensor pre_act_gates,
//     at::Tensor f_gate,
//     at::Tensor W,
//     at::Tensor R,
//     at::Tensor in_t,
//     at::Tensor h_tm1,
//     at::Tensor cell_tm1);


// std::vector<torch::Tensor> slstm_forward(
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor bias,
//     torch::Tensor old_h,
//     torch::Tensor old_cell) {
//   return slstm_cuda_forward(input, weights, bias, old_h, old_cell);
// }


// std::vector<torch::Tensor> slstm_backward(
//     at::Tensor grad_h,
//     at::Tensor grad_cell,
//     at::Tensor cell_t,
//     at::Tensor pre_act_gates,
//     at::Tensor f_gate,
//     at::Tensor W,
//     at::Tensor R,
//     at::Tensor in_t,
//     at::Tensor h_tm1,
//     at::Tensor cell_tm1) {

//   return slstm_cuda_backward(
//       grad_h,
//       grad_cell,
//       new_cell,
//       input_gate,
//       output_gate,
//       candidate_cell,
//       X,
//       gate_weights,
//       weights);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &slstm_forward, "SLSTM forward (CUDA)");
  m.def("backward", &slstm_backward, "SLSTM backward (CUDA)");
}
