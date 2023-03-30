import torch as T

# data
X = T.tensor([1, 2, 3, 4], dtype=T.float32)
y = T.tensor([2, 4, 6, 8], dtype=T.float32)

# model
w = T.tensor(0.0, dtype=T.float32, requires_grad=True)


def forward(x):
    return w * x


# loss: MSE
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')


# trainig
n_epoch = 200
lr = T.tensor(5e-3)
report_every = 10
for i_epoch in range(1, n_epoch+1):
    # forward
    y_pred = forward(X)
    # calc loss
    l = loss(y, y_pred)
    # calc grad
    l.backward()
    # back prob
    with T.no_grad():
        w -= lr * w.grad
    # reset gradient accum
    assert w.grad
    w.grad.zero_()
    # report
    if i_epoch % report_every == 0:
        print(f'Epoch {i_epoch}: {w=:.3f} {l=:.8f}')


print(f'Prediction after training: f(5) = {forward(5):.3f}')
