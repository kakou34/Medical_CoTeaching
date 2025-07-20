import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

# =========================
# Config
# =========================
noise_csv_path = "/home/nali1/noisy-label-learning-for-dementia-diagnosis/co_teaching/results/loss_per_sample_epoch_sym_50_1.csv"
root_dir = "./data"
batch_size = 128
epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load CIFAR-10
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

base_train = CIFAR10(root=root_dir, train=True, download=True)
clean_labels = base_train.targets

# =========================
# Generate noisy labels using noise status map
# =========================
noise_df = pd.read_csv(noise_csv_path)
noise_map = dict(zip(noise_df["Index"], noise_df["Noise_Status"]))
np.random.seed(42)
noisy_labels = clean_labels.copy()

for idx, status in noise_map.items():
    if status == 0:
        current = noisy_labels[idx]
        options = list(set(range(10)) - {current})
        noisy_labels[idx] = np.random.choice(options)

# =========================
# Custom Dataset
# =========================
class CIFAR10WithNoise(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

train_dataset = CIFAR10WithNoise(base_train.data, noisy_labels, transform)
test_dataset = CIFAR10(root=root_dir, train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =========================
# CNN model (your code)
# =========================
from model import CNN  # assumes your CNN is saved as model.py in the same directory
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# =========================
# Training
# =========================
for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f} | Acc: {correct/total*100:.2f}%")

# =========================
# Evaluation
# =========================
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    return np.array(all_targets), np.array(all_preds)

y_true, y_pred = evaluate(model, test_loader)
print("\nTest Metrics:")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
print(f"F1 Macro: {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"F1 Micro: {f1_score(y_true, y_pred, average='micro'):.4f}")
print("sym 50")
