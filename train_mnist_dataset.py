import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

# 1. Dataset and Dataloader
# download training data from open datasets
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomAffine(
            degrees = 15,        # small rotations (+15, -15 degrees)
            translate=(0.1, 0.1)  # small moves on x,y
        ),
        transforms.ToTensor()
    ])
)

# download test data from open datasets
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X,y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# 2. Define CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional part
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=24, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(24)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels = 24, out_channels=36, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(36)
        
        # Fully connected part
        self.fc1 = nn.Linear(36 * 5 * 5, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x,1) # flatten all dimensions except batch

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
print(model)

learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()


# 3. Training loop
EPOCHS = 20
best_acc = 0.0
scheduler = StepLR(optimizer, step_size=10, gamma = 0.1)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_dataset)
    train_acc = correct / total

    # evaluation test
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

        test_acc = test_correct / test_total
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f} | "
              f"Train acc: {train_acc:.4f} | "
              f"Test acc: {test_acc:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "mnist_cnn_best.pth")
            print(f"New best model saved with test acc ={best_acc: .4f}")

    scheduler.step()


