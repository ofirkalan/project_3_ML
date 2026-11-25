"""
HW3 - Task 1: an MLP and a CNN that classify CIFAR-10 images.
Machine Learning & Neural Networks for Neuroscience, Bar-Ilan University.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

# =============================================================================
# 1. Hyperparameters - change these and rerun to see what happens
# =============================================================================
SEED = 0                        # fixes the random numbers so a run can be repeated
VAL_SIZE = 5000                 # training images held out for validation
BATCH_SIZE = 128                # images per gradient step

# MLP
MLP_HIDDEN_SIZES = [1024, 512]  # one number per hidden layer = neurons in that layer
MLP_LEARNING_RATE = 5e-4
MLP_EPOCHS = 20

# CNN
CNN_CHANNELS = [32, 64, 128]    # one number per conv layer = filters in that layer
CNN_KERNEL_SIZE = 3             # filter is KERNEL x KERNEL pixels (odd number)
CNN_FC_SIZE = 256               # neurons in the fully connected hidden layer
CNN_LEARNING_RATE = 1e-3
CNN_EPOCHS = 20

# =============================================================================
# 2. Setup
# =============================================================================
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# =============================================================================
# 3. Data: download, split into train / validation / test, normalize
# =============================================================================
DATA_DIR = Path(__file__).parent / "data"
train_raw = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
test_raw = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)


def images_to_tensor(images):
    # images: uint8 array (N, 32, 32, 3) with values 0..255
    x = torch.tensor(images, dtype=torch.float32) / 255   # values 0..1
    return x.permute(0, 3, 1, 2)                          # (N, 3, 32, 32)


x_all = images_to_tensor(train_raw.data)
y_all = torch.tensor(train_raw.targets)
x_test = images_to_tensor(test_raw.data)
y_test = torch.tensor(test_raw.targets)

# Random split of the 50,000 training images into train and validation
shuffled = torch.randperm(len(x_all))
val_idx, train_idx = shuffled[:VAL_SIZE], shuffled[VAL_SIZE:]
x_train, y_train = x_all[train_idx], y_all[train_idx]
x_val, y_val = x_all[val_idx], y_all[val_idx]

# Normalize every color channel to mean 0 and std 1.
# The mean and std come from the training images only.
mean = x_train.mean(dim=(0, 2, 3), keepdim=True)
std = x_train.std(dim=(0, 2, 3), keepdim=True)
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE)


# =============================================================================
# 4. Model 1: Multi-Layer Perceptron
# =============================================================================
class MLP(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers = [nn.Flatten()]                 # (3, 32, 32) image -> vector of 3072
        in_features = 3 * 32 * 32
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            in_features = hidden
        layers.append(nn.Linear(in_features, 10))   # one output score per class
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 5. Model 2: Convolutional Neural Network
# =============================================================================
class CNN(nn.Module):
    def __init__(self, channels, kernel_size, fc_size):
        super().__init__()
        blocks = []
        in_channels = 3                         # RGB
        for out_channels in channels:
            blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
            blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool2d(2))      # halves the height and the width
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)

        side = 32 // (2 ** len(channels))       # image side after all the poolings
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * side * side, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =============================================================================
# 6. Training and evaluation
# =============================================================================
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss, total_correct, total_images = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        scores = model(x)                # forward pass: (batch, 10) class scores
        loss = loss_fn(scores, y)        # how wrong the scores are

        optimizer.zero_grad()            # erase the gradients of the previous step
        loss.backward()                  # backpropagation: d(loss)/d(every weight)
        optimizer.step()                 # move every weight a small step downhill

        total_loss += loss.item() * len(y)
        total_correct += (scores.argmax(dim=1) == y).sum().item()
        total_images += len(y)
    return total_loss / total_images, total_correct / total_images


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, total_correct, total_images = 0.0, 0, 0
    with torch.no_grad():                # no gradients needed: nothing is learned here
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            loss = loss_fn(scores, y)
            total_loss += loss.item() * len(y)
            total_correct += (scores.argmax(dim=1) == y).sum().item()
            total_images += len(y)
    return total_loss / total_images, total_correct / total_images


def train_model(name, model, learning_rate, epochs):
    print(f"\n===== {name} =====")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        print(f"epoch {epoch:2d} | train loss {train_loss:.3f} acc {train_acc:.3f}"
              f" | val loss {val_loss:.3f} acc {val_acc:.3f}")

    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print(f"{name} test accuracy: {test_acc:.3f}")


# =============================================================================
# 7. Run
# =============================================================================
if __name__ == "__main__":
    mlp = MLP(MLP_HIDDEN_SIZES)
    train_model("MLP", mlp, MLP_LEARNING_RATE, MLP_EPOCHS)

    cnn = CNN(CNN_CHANNELS, CNN_KERNEL_SIZE, CNN_FC_SIZE)
    train_model("CNN", cnn, CNN_LEARNING_RATE, CNN_EPOCHS)
