import time

import torch
from torch.nn import Linear, ReLU, Sequential

INPUT_DIM = 10000
OUTPUT_DIM = 10
HIDDEN_DIM = 512
HIDDEN_N = 1000
BATCH_SIZE = 512
EPOCH_N = 100

# Define the neural network model


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Sequential()
        self.net.append(torch.nn.Linear(INPUT_DIM, HIDDEN_DIM))
        self.net.append(ReLU())
        for _ in range(HIDDEN_N):
            self.net.append(Linear(HIDDEN_DIM, HIDDEN_DIM))
            self.net.append(ReLU())
        self.net.append(torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM))

    def forward(self, x):
        return self.net(x)


def speedtest(device):
    # Generate some random input data
    input_data = torch.randn(BATCH_SIZE, INPUT_DIM).to(device)

    # Create the model and move it to the specified device
    model = MyModel().to(device)

    # Run the model on the specified device
    print('Running the model on', device)
    start_time = time.time()
    for i in range(1, EPOCH_N+1):
        model(input_data)
        if i % 10:
            print('#', end='', flush=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('\nElapsed time:', elapsed_time)


# Test the model on the CPU
if input('Test on cpu? y/[N] ').upper() == 'Y':
    speedtest('cpu')

# Test the model on the GPU
if input('Test on cuda? [Y]/n ').upper() != 'N':
    speedtest('cuda')
