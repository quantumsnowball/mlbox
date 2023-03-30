import numpy as np

# data
X = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)

# model
w = 0.0


def forward(x):
    return w * x


# loss: MSE
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()


# l = 1/N (wx-y)^2
# dl/dw = 1/N 2(wx-y) x
def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred-y).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')


# trainig
n_epoch = 200
lr = 1e-3
report_every = 10
for i_epoch in range(n_epoch):
    # forward
    y_pred = forward(X)
    # calc loss
    l = loss(y, y_pred)
    # calc grad
    dw = gradient(X, y, y_pred)
    # back prob
    w -= lr * dw
    # report
    if i_epoch % report_every == 0:
        print(f'Epoch {i_epoch}: {w=:.3f} {l=:.8f}')


print(f'Prediction after training: f(5) = {forward(5):.3f}')
