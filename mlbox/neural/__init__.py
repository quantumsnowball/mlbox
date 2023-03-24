from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn
from typing_extensions import override


class NeuralNetwork(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...


class FullyConnected(NeuralNetwork):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 *args: Any,
                 n_hidden_layers: int = 2,
                 hidden_dim: int = 512,
                 Activation: type[nn.Module] = nn.ReLU,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        input_layer = nn.Linear(input_dim, hidden_dim)
        hidden_layers = [module
                         for _ in range(n_hidden_layers)
                         for module in [nn.Linear(hidden_dim, hidden_dim),
                                        Activation(), ]]
        output_layer = nn.Linear(hidden_dim, output_dim)
        self._network = nn.Sequential(
            input_layer,
            Activation(),
            *hidden_layers,
            output_layer,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        output: Tensor = self._network(x)
        return output
