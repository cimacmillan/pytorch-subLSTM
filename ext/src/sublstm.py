import math
import torch
import sublstm_cpp


# Torch autograd wrapper

class SUBLSTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, initial_h, initial_c, W, R, bias):
        outputs = sublstm_cpp.forward(input, W, R, bias, initial_h, initial_c)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [W] + [R] + [input] + [initial_h] + [initial_c]
        ctx.save_for_backward(*variables)
        return (new_h, new_cell)

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = sublstm_cpp.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
        grad_input, grad_h_tm1, grad_cell_tm1, grad_W, grad_R, grad_bias = outputs
        return grad_input, grad_h_tm1, grad_cell_tm1, grad_W, grad_R, grad_bias


class SUBLSTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(SUBLSTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.W = torch.nn.Parameter(
            torch.empty(4 * state_size, input_features))
        self.R = torch.nn.Parameter(
            torch.empty(4 * state_size, state_size))
        self.bias = torch.nn.Parameter(torch.empty(4 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return SUBLSTMFunction.apply(input, *state, self.W, self.R, self.bias)


