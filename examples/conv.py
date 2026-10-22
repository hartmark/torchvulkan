import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    """A ResNet for 32x32 images: three stages of residual blocks at widths ``width``, ``2 * width``
    and ``4 * width``, then global average pooling.

    ``width`` sets the channel count; ``benchmark/benchmark.py resnet`` sweeps it to scale the model up.
    """

    def __init__(self, width: int = 32, blocks_per_stage: int = 2, classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        layers, in_channels = [], width
        for stage, channels in enumerate((width, 2 * width, 4 * width)):
            for block in range(blocks_per_stage):
                stride = 2 if stage > 0 and block == 0 else 1
                layers.append(BasicBlock(in_channels, channels, stride))
                in_channels = channels
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(in_channels, classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


def main():
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torchvulkan

    batch_size = 128
    epochs = 2
    device = torch.device("vulkan" if torchvulkan.is_available() else "cpu")

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_loader = DataLoader(datasets.CIFAR10("./data", train=True, download=True, transform=train_transform), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10("./data", train=False, download=True, transform=test_transform), batch_size=512)

    model = ResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=epochs * len(train_loader))

    for epoch in range(epochs):
        model.train()
        for batch, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            loss = criterion(model(data), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch % 20 == 0:
                print(f"\repoch {epoch + 1}/{epochs} | batch {batch + 1}/{len(train_loader)} | loss {loss.item():.4f}", end="")

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, targets in test_loader:
                correct += (model(data.to(device)).argmax(dim=1) == targets.to(device)).sum().item()
        print(f"\repoch {epoch + 1}/{epochs} | test accuracy {100 * correct / len(test_loader.dataset):.2f}%" + " " * 20)


if __name__ == "__main__":
    main()
