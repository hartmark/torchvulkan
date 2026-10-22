import torch
import torch.nn as nn


class MLP(nn.Module):
    """A standard multi-layer perceptron classifier for 28x28 images.

    ``hidden`` sets the width of every hidden layer and ``depth`` how many there are;
    ``benchmark/benchmark.py mlp`` sweeps ``hidden`` to scale the model up.
    """

    def __init__(self, hidden: int = 512, depth: int = 3, in_features: int = 28 * 28, classes: int = 10, dropout: float = 0.1):
        super().__init__()
        layers = [nn.Flatten()]
        width = in_features
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        layers.append(nn.Linear(width, classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torchvulkan

    batch_size = 128
    epochs = 3
    device = torch.device("vulkan" if torchvulkan.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.MNIST("./data", train=False, download=True, transform=transform), batch_size=1024)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    for epoch in range(epochs):
        model.train()
        for batch, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            loss = criterion(model(data), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 50 == 0:
                print(f"\repoch {epoch + 1}/{epochs} | batch {batch + 1}/{len(train_loader)} | loss {loss.item():.4f}", end="")

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, targets in test_loader:
                correct += (model(data.to(device)).argmax(dim=1) == targets.to(device)).sum().item()
        print(f"\repoch {epoch + 1}/{epochs} | test accuracy {100 * correct / len(test_loader.dataset):.2f}%" + " " * 20)


if __name__ == "__main__":
    main()
