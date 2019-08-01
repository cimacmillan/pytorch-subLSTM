#include <torch/extension.h>
#include <iostream>
#include <vector>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<at::Tensor> sLSTM_cell_forward(
        at::Tensor input,
        at::Tensor W,
        at::Tensor R,
        at::Tensor bias,
        at::Tensor h_tm1,
        at::Tensor c_tm1) {

    auto pre_act_gates = at::add((at::mm(input, W.t()) + at::mm(h_tm1, R.t())), bias);
    auto gates = at::sigmoid(pre_act_gates).chunk(4, /*dim=*/1);

    auto in_gate = gates[0];
    auto f_gate = gates[1];
    auto z = gates[2];
    auto out_gate = gates[3];

    auto new_cell = f_gate * c_tm1 + z - in_gate;
    auto new_hidden = at::sigmoid(new_cell) - out_gate;

    return {new_hidden, new_cell, pre_act_gates, f_gate};
}

std::vector<at::Tensor> sLSTM_cell_backward(
        at::Tensor grad_h,
        at::Tensor grad_cell,
        at::Tensor cell_t,
        at::Tensor pre_act_gates,
        at::Tensor f_gate,
        at::Tensor W,
        at::Tensor R,
        at::Tensor in_t,
        at::Tensor h_tm1,
        at::Tensor cell_tm1) {

    auto d_cell = grad_h * d_sigmoid(cell_t) + grad_cell;
    // grads w.r.t. in_gate, f_gate, z and out_gate
    auto grads = torch::cat({-d_cell, d_cell * cell_tm1, d_cell, -grad_h}, /*dim*/1);

    grads *= d_sigmoid(pre_act_gates);
    auto t_grads = grads.t();

    // Compute the gradients
    auto grad_W = t_grads.mm(in_t);
    auto grad_R = t_grads.mm(h_tm1);
    auto grad_bias = grads.sum(/*dim=*/0, /*keepdim=*/true);

    // Compute errors
    auto grad_h_tm1 = grads.mm(R);
    auto grad_input = grads.mm(W);
    auto grad_cell_tm1 = grad_cell * f_gate;

    return {grad_input, grad_h_tm1, grad_cell_tm1, grad_W, grad_R, grad_bias};
}

// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sLSTM_cell_forward, "LLTM forward");
  m.def("backward", &sLSTM_cell_backward, "LLTM backward");
}