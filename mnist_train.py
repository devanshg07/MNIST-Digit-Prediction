import struct
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Paths to MNIST files
TRAIN_IMAGES = 'MNIST Dataset/train-images.idx3-ubyte'
TRAIN_LABELS = 'MNIST Dataset/train-labels.idx1-ubyte'
TEST_IMAGES = 'MNIST Dataset/t10k-images.idx3-ubyte'
TEST_LABELS = 'MNIST Dataset/t10k-labels.idx1-ubyte'

# Function to read images
def read_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
        return images

# Function to read labels
def read_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Load data
x_train = read_images(TRAIN_IMAGES)
y_train = read_labels(TRAIN_LABELS)
x_test = read_images(TEST_IMAGES)
y_test = read_labels(TEST_LABELS)

# Normalize and convert to tensors
x_train = torch.tensor(x_train / 255.0, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
x_test = torch.tensor(x_test / 255.0, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create datasets and loaders
train_ds = TensorDataset(x_train, y_train)
test_ds = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000)

# Define 2-layer neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}')

# Evaluation
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    print(f'Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    train(model, train_loader, criterion, optimizer, epochs=5)
    evaluate(model, test_loader) 