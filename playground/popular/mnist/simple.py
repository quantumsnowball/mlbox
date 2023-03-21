import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class DeepLearner:
    def __init__(self,
                 *,
                 Model: type[nn.Module],
                 DataSource: type[datasets.MNIST],
                 device='cuda',
                 batch_size: int = 8) -> None:
        self._training_data: Dataset = DataSource(
            root=".data",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        self._testing_data = DataSource(
            root=".data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
        self._train_dataloader = DataLoader(self._training_data,
                                            batch_size=batch_size)
        self._test_dataloader = DataLoader(self._testing_data,
                                           batch_size=batch_size)
        self._model: nn.Module = Model().to(device)
        self._loss_fn = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(),
                                          lr=1e-3)
        self._device = device

    def train(self, *, epochs: int = 5):
        # epoch
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")

            # batch
            size = len(self._train_dataloader.dataset)
            self._model.train()
            for batch, (X, y) in enumerate(self._train_dataloader):
                X, y = X.to(self._device), y.to(self._device)

                # Compute prediction error
                pred = self._model(X)
                loss = self._loss_fn(pred, y)

                # Backpropagation
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # test
            self.test()

        print("Done!")

    def test(self):
        size = len(self._test_dataloader.dataset)
        num_batches = len(self._test_dataloader)
        self._model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self._test_dataloader:
                X, y = X.to(self._device), y.to(self._device)
                pred = self._model(X)
                test_loss += self._loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


DeepLearner(
    Model=NeuralNetwork,
    DataSource=datasets.MNIST,
).train()
