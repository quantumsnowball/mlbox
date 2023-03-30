import torch as T
import torch.nn as nn
from torch.optim import optimizer

# data
X = T.tensor([1, 2, 3, 4], dtype=T.float32).unsqueeze(1)
y = T.tensor([2, 4, 6, 8], dtype=T.float32).unsqueeze(1)
X_test = T.tensor([5], dtype=T.float32)

# model
n_sample, n_feature = X.shape
model = nn.Linear(n_feature, n_feature)

# loss
loss = nn.MSELoss()


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')


# trainig
n_epoch = 500
lr = 5e-2
optimizer = T.optim.SGD(model.parameters(), lr=lr)
report_every = 10
for i_epoch in range(1, n_epoch+1):
    # forward
    y_pred = model(X)
    # calc loss
    l = loss(y, y_pred)
    # calc grad
    l.backward()
    # back prob
    optimizer.step()
    # reset gradient accum
    optimizer.zero_grad()
    # report
    if i_epoch % report_every == 0:
        w, b = model.parameters()
        print(f'Epoch {i_epoch}: {w[0][0].item()=:.3f} {l=:.8f}')


print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
