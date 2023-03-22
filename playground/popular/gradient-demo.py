import torch


def custom():
    x = torch.ones(5)
    w = torch.randn(5, 1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    y = torch.tensor(.07)
    z = torch.nn.functional.sigmoid(torch.matmul(x, w) + b)

    loss = -z * y
    breakpoint()
    return loss


def demo_derv(w1: float, w2: float, ):
    w1T = torch.tensor([float(w1), ], requires_grad=True)
    w2T = torch.tensor([float(w2), ], requires_grad=True)
    lossT = torch.log(3*w1T**2 + 5*w1T*w2T)
    print(f'z = {lossT}')
    lossT.backward()
    print(f'x.grad = {w1T.grad}')
    print(f'y.grad = {w2T.grad}')


demo_derv(3.0, 2.0)
demo_derv(2.0, 3.0)
