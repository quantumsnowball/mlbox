import torch
from torch import Tensor, nn


def fn(x: Tensor):
    return x**2 + 3


X = torch.tensor([
    [2],
    [3],
    [4],
], dtype=torch.float)

y = fn(X)

model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for i in range(50000):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(f'loss: {loss:>7f}')


with torch.no_grad():
    print(model(X))
    print(y)
