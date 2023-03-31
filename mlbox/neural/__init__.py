from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn
from typing_extensions import override


class NeuralNetwork(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        ...


class FullyConnected(NeuralNetwork):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 *args: Any,
                 n_hidden_layers: int = 2,
                 hidden_dim: int = 32,
                 Activation: type[nn.Module] = nn.ReLU,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # sequential module
        self._net = nn.Sequential()
        # input layer
        self._net.append(nn.Linear(input_dim, hidden_dim))
        self._net.append(Activation())
        # hidden layers
        for _ in range(n_hidden_layers):
            self._net.append(nn.Linear(hidden_dim, hidden_dim))
            self._net.append(Activation())
        # output layer
        self._net.append(nn.Linear(hidden_dim, output_dim))

    @override
    def forward(self, X: Tensor) -> Tensor:
        y_hat: Tensor = self._net(X)
        return y_hat
