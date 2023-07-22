from torch.utils.data import DataLoader
from auto_encoder import AutoEncoder
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import torch

device = torch.device('mps')
EPOCH = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_datasets = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_dataloader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True)

test_datasets = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_dataloader = DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=True)

model = AutoEncoder().to(device)
critention = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCH):
    print("- " * 20)

    model.train()
    train_loss, valid_loss = 0.0, 0.0
    for images, _ in train_dataloader:
        images = images.to(device)
        labels = images.to(device)

        output = model(images)
        output = output.view(BATCH_SIZE, 1, 28, 28)

        optimizer.zero_grad()
        loss = critention(output, labels)
        loss.backward()

        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss = train_loss / len(train_dataloader.dataset)
    print(f"[TRAIN] Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}")

    model.eval()
    with torch.no_grad():
        for images, _ in test_dataloader:
            images = images.to(device)
            labels = images.to(device)

            output = model(images)
            output = output.view(-1, 1, 28, 28)

            loss = critention(output, labels)
            valid_loss += loss.item() * images.size(0)
        valid_loss = valid_loss / len(test_dataloader.dataset)
        print(f"[VALID] Epoch: {epoch + 1}, Validation Loss: {valid_loss:.4f}")

print("Finished Training\n[INFO] Saving Model...")
torch.save(model.state_dict(), 'model.pth')
print("Finished Saving Model\nExiting...")

