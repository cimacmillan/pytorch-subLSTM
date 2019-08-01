
import time
import torch
from sublstm import *
from torch.nn import LSTM

USE_CUDA = False
device = torch.device("cuda") if USE_CUDA else torch.device("cpu")

n_layers = 1
# timesteps = 784
# trials = 1000
trials = 786000
batch_size = 16
input_features = 32
state_size = 128


def run(model):
    forward = 0
    backward = 0
    for _ in range(trials):
        start = time.time()

        (new_h, new_C) = model(X, (h, C))
        if USE_CUDA:
            torch.cuda.synchronize()

        forward += time.time() - start
        start = time.time()

        (new_h.sum() + new_C.sum()).backward()

        if USE_CUDA:
            torch.cuda.synchronize()

        backward += time.time() - start

    return forward, backward

X = torch.randn(batch_size, input_features, device=device)
h = torch.randn(batch_size, state_size, device=device)
C = torch.randn(batch_size, state_size, device=device)

forward, backward = run(SUBLSTM(input_features, state_size).to(device))
print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))
# forward, backward = run(LSTM(input_features, state_size).to(device))
# print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))