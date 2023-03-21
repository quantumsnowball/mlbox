import torch
from torch import Tensor, nn


X = torch.rand(1000, 3)

def irregular_fn(x: Tensor) -> Tensor:
    if x[0] < 0.5 and x[1] > 0.5 and x[2] < 0.5:
        return x[1]*5
    return torch.tensor(0.0)

y = torch.tensor([irregular_fn(x) for x in X], dtype=torch.float)

model = nn.Sequential(
    nn.Linear(3, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# loss_fn = nn.MSELoss()

for i in range(20000):
    pred = model(X)
    loss = -(y*pred).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 200 == 0:
        print(f'loss: {loss:>7f}')


with torch.no_grad():
    print(model(X)[:10])
    print(y[:10])
