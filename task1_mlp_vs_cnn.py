"""
HW3 - Task 1: an MLP and a CNN that classify CIFAR-10 images.
Machine Learning & Neural Networks for Neuroscience, Bar-Ilan University.

What the script does, in order:
  1. Defines the hyperparameters of both networks (edit them at the top).
  2. Downloads CIFAR-10 and splits it into train / validation / test sets.
  3. Builds a Multi-Layer Perceptron (MLP) and a Convolutional Network (CNN).
  4. Trains each network and prints its loss and accuracy after every epoch.

Run:  python task1_mlp_vs_cnn.py
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

# =============================================================================
# 1. Hyperparameters - change these and rerun to see what happens
# =============================================================================
# Shared by both networks
SEED = 0                        # seed of the random generator: same seed = same run
VAL_SIZE = 5000                 # training images held out for validation (not learned from)
BATCH_SIZE = 128                # images the network sees before each weight update

# MLP
MLP_HIDDEN_SIZES = [1024, 512]  # neurons in each hidden layer; list length = number of hidden layers
MLP_LEARNING_RATE = 5e-4        # size of each weight update
MLP_EPOCHS = 20                 # full passes over the training set

# CNN
CNN_CHANNELS = [32, 64, 128]    # filters in each conv layer; list length = number of conv layers
CNN_KERNEL_SIZE = 3             # each filter covers KERNEL x KERNEL pixels (use an odd number)
CNN_FC_SIZE = 256               # neurons in the fully connected hidden layer after the conv layers
CNN_LEARNING_RATE = 1e-3        # size of each weight update
CNN_EPOCHS = 20                 # full passes over the training set

# =============================================================================
# 2. Setup
# =============================================================================
torch.manual_seed(SEED)

# Use the GPU when there is one, otherwise the CPU. The same code runs on both.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# =============================================================================
# 3. Data: download, split into train / validation / test, normalize
# =============================================================================
# The dataset is downloaded once (about 170 MB) into a "data" folder next to
# this script. Later runs find it there and skip the download.
DATA_DIR = Path(__file__).parent / "data"
train_raw = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)    # 50,000 images
test_raw = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)    # 10,000 images


def images_to_tensor(images):
    """Convert raw images to the tensor format PyTorch layers expect.

    images: uint8 array of shape (N, 32, 32, 3) = (image, height, width, color),
            values 0..255.
    Returns a float tensor of shape (N, 3, 32, 32) = (image, color, height, width),
            values 0..1.
    """
    x = torch.tensor(images, dtype=torch.float32) / 255
    return x.permute(0, 3, 1, 2)


x_all = images_to_tensor(train_raw.data)
y_all = torch.tensor(train_raw.targets)      # labels: integers 0..9, one per image
x_test = images_to_tensor(test_raw.data)
y_test = torch.tensor(test_raw.targets)

# Split the 50,000 training images at random: VAL_SIZE images for validation,
# the rest for training. The validation images are never used to update weights;
# they measure how well the network does on images it did not learn from.
shuffled = torch.randperm(len(x_all))
val_idx, train_idx = shuffled[:VAL_SIZE], shuffled[VAL_SIZE:]
x_train, y_train = x_all[train_idx], y_all[train_idx]
x_val, y_val = x_all[val_idx], y_all[val_idx]

# Normalize each color channel to mean 0 and standard deviation 1.
# The mean and std are computed from the training images only and then applied
# to all three sets, so nothing about the validation or test images leaks into
# training. dim=(0, 2, 3) averages over images, height and width and leaves one
# value per color channel.
mean = x_train.mean(dim=(0, 2, 3), keepdim=True)
std = x_train.std(dim=(0, 2, 3), keepdim=True)
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std

# A DataLoader hands out the data in batches of BATCH_SIZE images.
# shuffle=True reorders the training images at the start of every epoch.
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE)


# =============================================================================
# 4. Model 1: Multi-Layer Perceptron
# =============================================================================
class MLP(nn.Module):
    """Fully connected network: flatten -> [Linear -> ReLU] per hidden layer -> Linear.

    hidden_sizes: list with the number of neurons of each hidden layer,
                  e.g. [1024, 512] builds 3072 -> 1024 -> 512 -> 10.
    Input:  batch of images, shape (batch, 3, 32, 32).
    Output: one score per class (logits), shape (batch, 10).
    """

    def __init__(self, hidden_sizes):
        super().__init__()
        # Flatten turns each (3, 32, 32) image into one vector of 3072 numbers.
        # Every pixel becomes a separate input feature; its position is lost.
        layers = [nn.Flatten()]
        in_features = 3 * 32 * 32
        for hidden in hidden_sizes:
            # Linear: every neuron computes a weighted sum of all its inputs + bias.
            layers.append(nn.Linear(in_features, hidden))
            # ReLU: the nonlinearity max(0, x). Without it, stacked linear
            # layers would collapse into a single linear layer.
            layers.append(nn.ReLU())
            in_features = hidden                    # this layer's output feeds the next one
        # Output layer: 10 scores, one per class. No softmax here because
        # the loss function (CrossEntropyLoss) applies it internally.
        layers.append(nn.Linear(in_features, 10))
        self.net = nn.Sequential(*layers)           # chain the layers in order

    def forward(self, x):
        """Forward pass: run the batch through all the layers."""
        return self.net(x)


# =============================================================================
# 5. Model 2: Convolutional Neural Network
# =============================================================================
class CNN(nn.Module):
    """Convolutional network: [Conv -> BatchNorm -> ReLU -> MaxPool] per conv layer,
    then flatten -> Linear -> ReLU -> Linear.

    channels:    list with the number of filters of each conv layer.
    kernel_size: side of the square filter, in pixels.
    fc_size:     neurons in the fully connected hidden layer.
    Input:  batch of images, shape (batch, 3, 32, 32).
    Output: one score per class (logits), shape (batch, 10).
    """

    def __init__(self, channels, kernel_size, fc_size):
        super().__init__()

        # ---- Feature extractor: the convolutional blocks ----
        blocks = []
        in_channels = 3                             # the input image has 3 color channels (RGB)
        for out_channels in channels:
            # Conv2d: out_channels filters, each kernel_size x kernel_size x in_channels,
            # slide over the image and produce one activation map per filter.
            # padding = kernel_size // 2 adds a border of zeros so the output
            # keeps the same height and width as the input.
            blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
            # BatchNorm2d: normalizes each activation map over the batch
            # (mean 0, std 1, then a learned scale and shift). Speeds up training.
            blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(nn.ReLU())
            # MaxPool2d(2): keeps the largest value in every 2x2 window,
            # which halves the height and the width.
            blocks.append(nn.MaxPool2d(2))
            in_channels = out_channels              # this layer's maps are the next layer's input
        self.features = nn.Sequential(*blocks)

        # ---- Classifier: fully connected layers on top of the features ----
        # Each pooling halves the image side: 32 -> 16 -> 8 -> 4 for three conv layers.
        side = 32 // (2 ** len(channels))
        self.classifier = nn.Sequential(
            nn.Flatten(),                                       # (channels, side, side) -> vector
            nn.Linear(in_channels * side * side, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, 10),                             # 10 class scores
        )

    def forward(self, x):
        """Forward pass: extract features with the conv blocks, then classify them."""
        return self.classifier(self.features(x))


# =============================================================================
# 6. Training and evaluation
# =============================================================================
def train_one_epoch(model, loader, loss_fn, optimizer):
    """Train the model for one full pass over the data in `loader`.

    Returns the mean loss and the accuracy (fraction of correct predictions)
    over that pass.
    """
    model.train()                        # training mode: BatchNorm uses the statistics of each batch
    total_loss, total_correct, total_images = 0.0, 0, 0
    for x, y in loader:                  # x: batch of images, y: their correct labels
        x, y = x.to(device), y.to(device)

        # 1. Forward pass: compute the 10 class scores of every image in the batch.
        scores = model(x)
        # 2. Loss: one number that says how wrong the scores are (mean over the batch).
        loss = loss_fn(scores, y)

        # 3. Erase the gradients left over from the previous batch
        #    (PyTorch accumulates gradients unless told otherwise).
        optimizer.zero_grad()
        # 4. Backpropagation: compute d(loss)/d(weight) for every weight
        #    with the chain rule.
        loss.backward()
        # 5. Update: move every weight a small step in the direction that
        #    lowers the loss. The step size is set by the learning rate.
        optimizer.step()

        # Bookkeeping for the printed statistics.
        total_loss += loss.item() * len(y)                          # loss.item() is the batch mean
        total_correct += (scores.argmax(dim=1) == y).sum().item()   # prediction = class with the highest score
        total_images += len(y)
    return total_loss / total_images, total_correct / total_images


def evaluate(model, loader, loss_fn):
    """Measure the mean loss and the accuracy of the model on `loader`.

    Nothing is learned here: there is no backpropagation and no weight update.
    """
    model.eval()                         # evaluation mode: BatchNorm uses its running averages
    total_loss, total_correct, total_images = 0.0, 0, 0
    with torch.no_grad():                # do not track gradients: saves time and memory
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            loss = loss_fn(scores, y)
            total_loss += loss.item() * len(y)
            total_correct += (scores.argmax(dim=1) == y).sum().item()
            total_images += len(y)
    return total_loss / total_images, total_correct / total_images


def train_model(name, model, learning_rate, epochs):
    """Train `model` for `epochs` epochs and print its progress.

    After every epoch prints the loss and accuracy on the training set and on
    the validation set. After the last epoch prints the test accuracy.
    """
    print(f"\n===== {name} =====")
    model.to(device)
    # Cross-entropy: turns the 10 scores into probabilities (softmax) and
    # penalizes a low probability for the correct class.
    loss_fn = nn.CrossEntropyLoss()
    # Adam: gradient descent with momentum and a separate adaptive step size
    # for every weight.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        print(f"epoch {epoch:2d} | train loss {train_loss:.3f} acc {train_acc:.3f}"
              f" | val loss {val_loss:.3f} acc {val_acc:.3f}")

    # The test set is used once, after training has finished.
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print(f"{name} test accuracy: {test_acc:.3f}")


# =============================================================================
# 7. Run: build each network from the hyperparameters above and train it
# =============================================================================
if __name__ == "__main__":
    mlp = MLP(MLP_HIDDEN_SIZES)
    train_model("MLP", mlp, MLP_LEARNING_RATE, MLP_EPOCHS)

    cnn = CNN(CNN_CHANNELS, CNN_KERNEL_SIZE, CNN_FC_SIZE)
    train_model("CNN", cnn, CNN_LEARNING_RATE, CNN_EPOCHS)
