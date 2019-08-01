# The subLSM architecture

## Motivation





## The subLSTM cell

Formally, the internal workings of a subLSTM cell can be defined as follows. Let $\mathbf{x}_t$ be the input, $\mathbf{h}_{t-1}$ the hidden activation at the previous time step,  $\mathbf{z}_t$ be the input to the cell and $\mathbf{i}_t$,$\mathbf{o}_t$, $\mathbf{f}_t$ the input, output and forget gates respectively we have:


$$
\begin{aligned}
\big[\mathbf{z}_t, \mathbf{i}_t, \mathbf{o}_t, \mathbf{f}_t\big] &= \sigma(\mathbf{W}\mathbf{x}_t + \mathbf{R}\mathbf{h}_{t-1} + \mathbf{b}) \\ 
\mathbf{c}_t &= \mathbf{c}_{t-1} \odot \mathbf{f}_t + \mathbf{z}_t - \mathbf{i}_t \\
\mathbf{h}_t &= \sigma(\mathbf{c}_t) - \mathbf{o}_t
\end{aligned}
$$

The output for this layer is the vector concatenation of both vectors: $[\mathbf{h}_t; \mathbf{c}_t]$, which carries the output of the network at time $t$ and the new memory state. 

When back-propagating we need to take into account errors coming from both values. Additionally, the gradient of the loss w.r.t. the hidden activation is a sum of the gradient of the loss at time $t$ (if any), and the errors coming from subsequent time steps.

The gradients with respect to each parameter are computed as follows. Given the error from the layer above and the next time step $\big[\Delta_t; \delta\mathbf{\hat{c}}_t\big]$, we apply the chain rule until we reach each parameter. Starting from the errors we have:
$$
\begin{aligned}
\delta\mathbf{h}_t &= \Delta_t \\

\mathbf{h}'_t &= \sigma'(\mathbf{c}_t)\mathbf{c}'_t - \mathbf{o}'_t \\
\mathbf{c}'_t &= \mathbf{c}_{t-1} \odot \mathbf{f}'_t+ \mathbf{z}'_t - \mathbf{i}'_t \\
\end{aligned}
$$
Which implies that the loss w.r.t. to the memory cell is:
$$
\begin{aligned}
\delta \mathbf{c}_t &= \delta\mathbf{h}_t\frac{d\mathbf{h}_t}{d\mathbf{c}_t} + \delta\hat{\mathbf{c}}_t \\
&= \delta\mathbf{h}_t\sigma'(\mathbf{c}_t)+ \delta\hat{\mathbf{c}}_t
\end{aligned}
$$
Thus deriving the loss w.r.t. each gate and the input to the cell we have:
$$
\begin{aligned}
\delta\overline{\mathbf{o}}_t &= -\delta\mathbf{h}_t\odot\sigma'(\overline{\mathbf{o}}_t) \\
\delta\overline{\mathbf{f}}_t &= \delta\mathbf{c}_t \odot \mathbf{c}_{t-1} \odot \sigma'(\overline{\mathbf{f}}_t) \\
\delta\overline{\mathbf{i}}_t &= -\delta\mathbf{c}_t \odot \sigma'(\overline{\mathbf{i}}_t) \\
\delta\overline{\mathbf{z}}_t &= \delta\mathbf{c}_t \odot \sigma'(\overline{\mathbf{z}}_t) \\
\end{aligned}
$$
And combining these with the derivative for the activations w.r.t. to the parameters we have for each one of them the following equations:
$$
\begin{aligned}
\bigg[\delta\mathbf{W}_o; \delta\mathbf{R}_o ; \delta\mathbf{b}_o\bigg] &=\bigg[-\Big(\delta \mathbf{h}_t \odot \sigma'(\mathbf{\overline{o}}_t)\Big)\cdot\mathbf{x}_{t}^T; -\Big(\delta \mathbf{h}_t \odot \sigma'(\mathbf{\overline{o}}_t)\Big)\cdot\mathbf{h}_{t-1}^T ; -\delta \mathbf{h}_t \odot \sigma'(\mathbf{\overline{o}}_t)\bigg] \\

\bigg[\delta\mathbf{W}_i; \delta\mathbf{R}_i ; \delta\mathbf{b}_i\bigg] &= \bigg[-\Big(\delta\mathbf{c}_t\odot\sigma'(\overline{\mathbf{i}}_t)\Big)\cdot\mathbf{x}^T_t; -\Big(\delta\mathbf{c}_t \odot \sigma'(\mathbf{\overline{i}}_t)\Big)\cdot\mathbf{h}_{t-1}^T ; - \delta \mathbf{c}_t \odot \sigma'(\mathbf{\overline{i}}_t)\bigg] \\

\bigg[\delta\mathbf{W}_f; \delta\mathbf{R}_f ; \delta\mathbf{b}_f\bigg] &= 
\bigg[\Big(\delta\mathbf{c}_t\odot\mathbf{c}_{t-1}\odot \sigma'(\overline{\mathbf{f}}_t)\Big)\cdot\mathbf{x}^T_t; \Big(\delta\mathbf{c}_t\odot\mathbf{c}_{t-1}\odot \sigma'(\overline{\mathbf{f}}_t)\Big)\cdot\mathbf{h}^T_{t-1}; \delta \mathbf{c}_t \odot\mathbf{c}_{t-1}\odot \sigma'(\mathbf{\overline{f}}_t) \bigg] \\

\bigg[\delta\mathbf{W}_z; \delta\mathbf{R}_z ; \delta\mathbf{b}_z\bigg] &= \bigg[ \Big(\delta\mathbf{c}_t\odot \sigma'(\overline{\mathbf{z}}_t)\Big) \cdot\mathbf{x}^T_t; \Big(\delta\mathbf{c}_t\odot \sigma'(\overline{\mathbf{z}}_t)\Big)\cdot\mathbf{h}^T_{t-1}; \delta \mathbf{c}_t \odot \sigma'(\mathbf{\overline{f}}_t) \bigg]
\end{aligned}
$$
Further details about how to propagate the errors and compute the gradients along with an implementation in PyTorch's autograd can be found in the appendix.

## Noisy gradient propagation in subLSTMs



# Appendix

## A1: Back-propagation in the subLSTM model

In order to do back-propagation through time (BPTT), we need to compute the gradient of the subLSTM cell for each parameter. Let $\delta v = \frac{d\mathcal{L}(y, f_\theta(\mathbf{x}))}{dv}$ be the derivative of the loss computed by the model with respect to some intermediate value $v$.  Applying the chain rule, the error coming from the next time step is: 
$$
\frac{d\mathcal{L}(y, f_\theta(\mathbf{x}))}{d\theta_{t+1}}= \bigg[\delta \mathbf{h}_t; \delta\mathbf{\hat{c}_t}\bigg]^T  = \Delta_t
$$
Thus we only need to apply the chain rule to this value for each parameter. Denoting anyone of the pre-activation values of $\mathbf{z}_t, \mathbf{i}_t, \mathbf{o}_t, \mathbf{f}_ t​$ as $\overline{\mathbf{y}}_t​$:
$$
\delta \overline{\mathbf{y}}_t  = \Delta_t\frac{d[\mathbf{h}_t;\mathbf{\hat{c}}]}{d{\mathbf{y}}_t}\frac{d\mathbf{y}_t}{d\overline{\mathbf{y}}}
$$
Where the last step of each addend follows these equations:
$$
\begin{aligned}
\mathbf{y}'_t &= \sigma'(\overline{\mathbf{y}}_t)\overline{\mathbf{y}}'_t \\
\overline{\mathbf{y}}_t &= \mathbf{W}_y\mathbf{x}_t + \mathbf{R}_y\mathbf{h}_{t-1} + \mathbf{b}_y
\end{aligned}
$$
In the case of the middle step, for the hidden unit activation we have:
$$
\mathbf{h}'_t = \sigma'(\mathbf{c}_t)\mathbf{c}'_t - \mathbf{o}'_t
$$
And for the memory cell:
$$
\begin{aligned}
\delta \mathbf{c}_t &= \delta\mathbf{h}_t\frac{d\mathbf{h}_t}{d\mathbf{c}_t} + \delta\hat{\mathbf{c}}_t \\
&= \delta\mathbf{h}_t\sigma'(\mathbf{c}_t) + \delta\hat{\mathbf{c}}_t \\
\mathbf{c}'_t &= \mathbf{c}_{t-1} \odot \mathbf{f}'_t+ \mathbf{z}'_t - \mathbf{i}'_t
\end{aligned}
$$
**NOTE**: since these are derivatives with respect to the inputs, when computing the gradients w.r.t. each parameter some terms will cancel out. For example, the gradient w.r.t to the weights of the output gate will not depend on the first term of $\mathbf{h}'_t$.

We can now obtain the gradient of the loss with respect to each parameter by substituting these quantities in turn, cancelling the relevant ones as we go. For the output gate we have:
$$
\begin{aligned}
\delta\mathbf{W}_o &= \delta\mathbf{h}_t\frac{d\mathbf{h}_t}{d\overline{\mathbf{o}}_t}\frac{d\overline{\mathbf{o}}_t}{d\mathbf{W}_0} \\
&= \bigg[\delta \mathbf{h}_t \odot (-\sigma'(\mathbf{\overline{o}}_t))\bigg] \cdot \mathbf{x}_{t}^T \\
&= -\bigg[\delta \mathbf{h}_t \odot \sigma'(\mathbf{\overline{o}}_t)\bigg]\cdot\mathbf{x}_{t}^T
\end{aligned}
$$
Analogously we have for the recurrent weight and the bias:
$$
\begin{aligned}
\delta\mathbf{R}_o &= \delta\mathbf{h}_t\frac{dh_t}{d\overline{\mathbf{o}}_t}\frac{d\overline{\mathbf{o}}_t}{d\mathbf{R}_o}= -\bigg[\delta \mathbf{h}_t \odot \sigma'(\mathbf{\overline{o}}_t)\bigg]\cdot\mathbf{h}_{t-1}^T \\

\delta\mathbf{b}_o &= \delta\mathbf{h}_t\frac{dh_t}{d\overline{\mathbf{o}}_t}\frac{d\overline{\mathbf{o}}_t}{d\mathbf{b}_o}= -\bigg[\delta \mathbf{h}_t \odot \sigma'(\mathbf{\overline{o}}_t)\bigg]
\end{aligned}
$$


For the input gate the error flows through the memory cell, so in this case we have:
$$
\begin{aligned}
\delta \mathbf{W}_i &= \delta\mathbf{c}_t\frac{d\mathbf{c}_t}{d\mathbf{i}_t}\frac{d\mathbf{i}_t}{d\overline{\mathbf{i}}_t}\frac{d\overline{\mathbf{i}}_t}{dW_i}\\
&= \bigg[\delta\mathbf{c}_t\odot(-1)\odot\sigma'(\overline{\mathbf{i}}_t)\bigg] \mathbf{x}^T_t \\
&= -\bigg[\delta\mathbf{c}_t\odot\sigma'(\overline{\mathbf{i}}_t)\bigg]\cdot\mathbf{x}^T_t
\end{aligned}
$$
And similarly for the other parameters we get:
$$
\begin{aligned}
\delta\mathbf{R}_i &= -\bigg[\delta\mathbf{c}_t \odot \sigma'(\mathbf{\overline{i}}_t)\bigg]\cdot\mathbf{h}_{t-1}^T \\

\delta\mathbf{b}_i &= -\delta \mathbf{c}_t \odot \sigma'(\mathbf{\overline{i}}_t)
\end{aligned}
$$
The forget gate proceeds similarly, we just need to keep in mind that the gradient of the memory cell with respect to it is different:
$$
\begin{aligned}\delta \mathbf{W}_f &= \delta\mathbf{c}_t\frac{d\mathbf{c}_t}{d\mathbf{f}_t}\frac{d\mathbf{f}_t}{d\overline{\mathbf{f}}_t}\frac{d\overline{\mathbf{f}}_t}{d\mathbf{W}_f}\\
&= \bigg[\delta\mathbf{c}_t\odot\mathbf{c}_{t-1}\odot \sigma'(\overline{\mathbf{f}}_t)\bigg]\cdot\mathbf{x}^T_t \\
\end{aligned}
$$
And proceeding the same way for the other parameters:
$$
\begin{aligned}
\delta \mathbf{R}_f &= \bigg[\delta\mathbf{c}_t\odot\mathbf{c}_{t-1}\odot \sigma'(\overline{\mathbf{f}}_t)\bigg]\cdot\mathbf{h}^T_{t-1} \\
\delta\mathbf{b}_i &= \delta \mathbf{c}_t \odot\mathbf{c}_{t-1}\odot \sigma'(\mathbf{\overline{f}}_t)
\end{aligned}
$$
And lastly for the input we have:
$$
\begin{aligned}
\delta \mathbf{W}_z
&= \bigg[\delta\mathbf{c}_t\odot \sigma'(\overline{\mathbf{z}}_t)\bigg]\cdot\mathbf{x}^T_t \\
\delta \mathbf{R}_z &= \bigg[\delta\mathbf{c}_t\odot \sigma'(\overline{\mathbf{z}}_t)\bigg]\cdot\mathbf{h}^T_{t-1} \\
\delta\mathbf{b}_i &= \delta \mathbf{c}_t \odot \sigma'(\mathbf{\overline{f}}_t)
\end{aligned}
$$
This gives us the following update rules for the gradients at time $t​$:
$$
\begin{aligned}
\bigg[\delta\mathbf{W}_o; \delta\mathbf{R}_o ; \delta\mathbf{b}_o\bigg] &=\bigg[-\Big(\delta \mathbf{h}_t \odot \sigma'(\mathbf{\overline{o}}_t)\Big)\cdot\mathbf{x}_{t}^T; -\Big(\delta \mathbf{h}_t \odot \sigma'(\mathbf{\overline{o}}_t)\Big)\cdot\mathbf{h}_{t-1}^T ; -\delta \mathbf{h}_t \odot \sigma'(\mathbf{\overline{o}}_t)\bigg] \\

\bigg[\delta\mathbf{W}_i; \delta\mathbf{R}_i ; \delta\mathbf{b}_i\bigg] &= \bigg[-\Big(\delta\mathbf{c}_t\odot\sigma'(\overline{\mathbf{i}}_t)\Big)\cdot\mathbf{x}^T_t; -\Big(\delta\mathbf{c}_t \odot \sigma'(\mathbf{\overline{i}}_t)\Big)\cdot\mathbf{h}_{t-1}^T ; - \delta \mathbf{c}_t \odot \sigma'(\mathbf{\overline{i}}_t)\bigg] \\

\bigg[\delta\mathbf{W}_f; \delta\mathbf{R}_f ; \delta\mathbf{b}_f\bigg] &= 
\bigg[\Big(\delta\mathbf{c}_t\odot\mathbf{c}_{t-1}\odot \sigma'(\overline{\mathbf{f}}_t)\Big)\cdot\mathbf{x}^T_t; \Big(\delta\mathbf{c}_t\odot\mathbf{c}_{t-1}\odot \sigma'(\overline{\mathbf{f}}_t)\Big)\cdot\mathbf{h}^T_{t-1}; \delta \mathbf{c}_t \odot\mathbf{c}_{t-1}\odot \sigma'(\mathbf{\overline{f}}_t) \bigg] \\

\bigg[\delta\mathbf{W}_z; \delta\mathbf{R}_z ; \delta\mathbf{b}_z\bigg] &= \bigg[ \Big(\delta\mathbf{c}_t\odot \sigma'(\overline{\mathbf{z}}_t)\Big) \cdot\mathbf{x}^T_t; \Big(\delta\mathbf{c}_t\odot \sigma'(\overline{\mathbf{z}}_t)\Big)\cdot\mathbf{h}^T_{t-1}; \delta \mathbf{c}_t \odot \sigma'(\mathbf{\overline{f}}_t) \bigg]
\end{aligned}
$$
Where we have concatenated the gradients for each gate. For the errors that need to be backpropagated to the previous time step, notice that for the hidden activations,  they appear in the gradients of the recurrent steps only, so we just need to multiply the factors preceding the activations by the corresponding transposed weights and add them together. 

The same applies for the gradient with respect to the input, which is necessary if we have several layers, but exchanging the relevant weights. Finally, the gradient with respect to the old memory cell, we observe that it only affects the current cell through its interaction with the forget gate. Thus we have:
$$
\begin{aligned}
\delta\mathbf{x}_t &= -\mathbf{W}^T_o\Big(\delta\mathbf{h}_t \odot \sigma'(\overline{\mathbf{o}}_t)\Big) -\mathbf{W}^T_i\Big(\delta\mathbf{c}_t \odot \sigma'(\overline{\mathbf{i}}_t)\Big) + \mathbf{W}^T_f\Big(\delta\mathbf{c}_t \odot \mathbf{c}_{t-1} \odot \sigma'(\overline{\mathbf{f}}_t)\Big) + \mathbf{W}^T_z\Big(\delta\mathbf{c}_t \odot \sigma'(\overline{\mathbf{z}}_t)\Big) \\

\delta\mathbf{h}_{t-1} &= -\mathbf{R}^T_o\Big(\delta\mathbf{h}_t \odot \sigma'(\overline{\mathbf{o}}_t)\Big) -\mathbf{R}^T_i\Big(\delta\mathbf{c}_t \odot \sigma'(\overline{\mathbf{i}}_t)\Big) + \mathbf{R}^T_f\Big(\delta\mathbf{c}_t \odot \mathbf{c}_{t-1} \odot \sigma'(\overline{\mathbf{f}}_t)\Big) + \mathbf{R}_z^T\Big(\delta\mathbf{c}_t \odot \sigma'(\overline{\mathbf{z}}_t)\Big) \\

\delta\mathbf{c}_{t-1} &= \delta\mathbf{c}_t \odot \mathbf{f}_t
\end{aligned}
$$
And we can use these to backpropagate the gradient to previous layers and time steps.

## A2: Implementing the subLSTM cell

When implementing these equations, we usually pass examples in batches which come in a design matrix $\mathbf{X}_t \in M(N, D)$. To take this into account we rewrite the formulas in matrix form as, so the outputs have a leading dimension of size $N$ as well:
$$
\begin{aligned}
\mathcal{G} = \big[\mathbf{Z}_t, \mathbf{I}_t, \mathbf{O}_t, \mathbf{F}_t\big] &= \sigma(\mathbf{X_t}\mathbf{W}^T_t + \mathbf{H}_{t-1}\mathbf{R}^T_t + \mathbf{I} \mathbf{b}_t) \\
\mathbf{C}_t &= \mathbf{C}_{t-1} \odot \mathbf{F}_t + \mathbf{Z}_t - \mathbf{I}_t \\
\mathbf{H}_t &= \sigma(\mathbf{C}_t) - \mathbf{O}_t
\end{aligned}
$$
We can rewrite the gradients w.r.t. the weights as a multiplication of matrices. With a slight abuse of notation we have:
$$
\begin{aligned}
\nabla\mathbf{W}^{(t)} &= \delta\mathbf{W} = \nabla\mathcal{G}^T\cdot\mathbf{X}_t \\
\nabla\mathbf{R}^{(t)} &= \delta\mathbf{R} = \nabla\mathcal{G}^T\cdot\mathbf{H}_{t-1} \\
\nabla\mathbf{b}^{(t)} &= \sum_{i=1}^N\nabla \mathcal{G}_i
\end{aligned}
$$
where $\nabla\mathcal{G} \in M(N, 4H)​$ is the gradient w.r.t. each of the gates for each input in the batch and the matrix $\mathbf{X}_t \in M(N, D + H)​$ is the design matrix of inputs in a batch, each one of which is a row vector that concatenates the transposed inputs and hidden activations from the previous time step. Thus we can do the computations efficiently by using matrix operations.

We can proceed in a similar fashion and define the gradients w.r.t. to the inputs as matrix multiplications:
$$
\begin{aligned}
\nabla\mathbf{X}_t &= \nabla\mathcal{G} \cdot \mathbf{W}\\
\nabla\mathbf{H}_{t-1} &= \nabla\mathcal{G} \cdot \mathbf{R} \\
\nabla\mathbf{C}_{t-1} &= \nabla\mathbf{C}_t \odot \mathbf{F}_t
\end{aligned}
$$
where again $\nabla\mathcal{G} \in M(N, 4 H)​$ and $\mathbf{W} \in M(4H, D)​$, $\mathbf{R} \in M(4H, H)​$ and $\nabla\mathbf{C}_t, \mathbf{F}_t \in M(N, H)​$ with $\mathbf{F}_t​$ being the per instance forget gate activation matrix.

### SubLTM cell code

The is a sample code for the forward and backward passes of the subLSTM cell in C++ using PyTorch's ATen library:

```c++
#include <iostream>
#include <vector>
#include <torch/extension.h>

at::Tensor d_sigmoid(at::Tensor z){
    auto s = at::sigmoid(z);
    return (1 - s) * s;
}

std::vector<at::Tensor> sLSTM_cell_forward(
        at::Tensor input,
        at::Tensor W,
        at::Tensor R,
        at::Tensor bi,
        at::Tensor bh,
        at::Tensor h_tm1,
        at::Tensor c_tm1) {

    auto pre_act_gates = at::addmm(bi, input, W.t()) + at::addmm(bh, h_tm1, R.t());
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
```



