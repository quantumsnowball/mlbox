import torch as T
import torch.nn as nn

X_list = [3, 4, 5]
y_list = [2*x for x in X_list]
for X_val, y_val in zip(X_list, y_list):
    m = T.tensor(3.0, requires_grad=True)
    X = T.tensor(X_val)
    y = T.tensor(y_val)
    out = m*m*X
    out.backward()
    with T.no_grad():
        print(f'{X=} {m.grad=} {2*m*X=}')
        assert m.grad == 2*m*X
