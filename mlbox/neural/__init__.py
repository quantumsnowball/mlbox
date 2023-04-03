from typing import Any

from torch.nn import Linear, Module, ReLU, Sequential


class FullyConnected(Sequential):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 *args: Any,
                 n_hidden_layers: int = 2,
                 hidden_dim: int = 32,
                 Activation: type[Module] = ReLU,
                 OutputActivation: type[Module] | None = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # input layer
        self.append(Linear(input_dim, hidden_dim))
        self.append(Activation())
        # hidden layers
        for _ in range(n_hidden_layers):
            self.append(Linear(hidden_dim, hidden_dim))
            self.append(Activation())
        # output layer
        self.append(Linear(hidden_dim, output_dim))
        if OutputActivation:
            self.append(OutputActivation())
