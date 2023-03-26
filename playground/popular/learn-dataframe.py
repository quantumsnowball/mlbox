import asyncio

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from trbox.market.yahoo.historical.windows import fetch_sqlite


# make dataset
def ohlcv(*args, **kwargs):
    return asyncio.run(fetch_sqlite(*args, **kwargs))


class Ohlcv(Dataset):
    def __init__(self) -> None:
        super().__init__()
        df = ohlcv('SPY', '1d', '2015-01-01', '2023-01-01')
        df['chg5d'] = df['Close'].pct_change(5).shift(-5)
        df['pnl-ratio'] = df['Close'].rank(pct=True)
        df = df.dropna()
        self._df = df

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        return (
            torch.tensor([self._df['pnl-ratio'][index], ], dtype=torch.float),
            torch.tensor(self._df['chg5d'][index], dtype=torch.float),
        )


data = Ohlcv()
train_loader = DataLoader(data, batch_size=64)
test_loader = DataLoader(data, batch_size=1)

# nn
device = 'cuda'
model = nn.Sequential(
    nn.Linear(1, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# training
n_epoch = 10
for t in range(n_epoch):
    print(f"Epoch {t+1}\n-------------------------------")

    for X, y in train_loader:
        model.train()

        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            action = model(X)
            profit = (action * y).mean()
            loss = -profit

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss = loss.item()
                print(f'loss: {loss:>7f}')

# eval
model.eval()
with torch.no_grad():
    for i, (X_, y_) in enumerate(test_loader):
        X_, y_ = X_.to(device), y_.to(device)
        action = model(X_)
        profit = (action * y_).mean()
        print(f'Chg: {y_}, Action: {action}, Profit: {profit}')
        if i >= 5:
            break
