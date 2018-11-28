#!/bin/bash

cd ..
source ../../.venv/bin/activate

python run.py --model subLSTM --nlayers 1 --nhid 50 \
    --optim rmsprop --lr 1e-4 --l2-norm 0.1 --epochs 2 --batch-size 50 \
    --seed 18092 --cuda --log-interval 50 --data MNIST --save results
