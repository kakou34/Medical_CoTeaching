import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImage, Resize, NormalizeIntensity
from monai.networks.nets import resnet18
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


# Config

root_dir = ""
csv_path = ""
test_csv = ""
noise_map_csv = ""
num_classes = 2
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16


# Load noise map

noise_df = pd.read_csv(noise_map_csv)
noisy_label_map = dict(zip(noise_df["PTID"], noise_df["Noisy_Label"]))


# Transforms

transform = Compose([
    LoadImage(ensure_channel_first=True),
    Resize((96, 116, 96)),
    NormalizeIntensity(nonzero=True, channel_wise=True),
])


# Dataset

class ADNIBaselineDataset(Dataset):
    def __init__(self, csv_file, root, transform, label_map=None):
        df = pd.read_csv(csv_file)
        self.root = root
        self.transform = transform
        self.label_map = label_map or {}

        self.ptids = df["PTID"].tolist()
        self.label_lookup = dict(zip(df["PTID"], df["binaryLabel"]))

        if label_map:
            self.ptids = [p for p in self.ptids if p in label_map]

    def __len__(self):
        return len(self.ptids)

    def __getitem__(self, idx):
        ptid = self.ptids[idx]
        img_path = os.path.join(self.root, ptid + ".nii.gz")
        img = self.transform(img_path)

        if self.label_map and ptid in self.label_map:
            label = self.label_map[ptid]
        else:
            label = self.label_lookup[ptid]

        return {"img": img, "label": label, "id": ptid}


# Evaluation

def evaluate(model, dataloader, device, loss_function, return_probs=False):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    total_loss, steps = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch["img"].to(device), batch["label"].to(device)
            outputs = model(inputs)

            loss = loss_function(outputs, targets)
            total_loss += loss.item()
            steps += 1

            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / steps
    y_true, y_pred, y_prob = np.array(all_targets), np.array(all_preds), np.array(all_probs)

    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except:
        auc = float("nan")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "roc_auc": auc,
        "loss": avg_loss
    }

    if return_probs:
        return metrics, y_prob
    else:
        return metrics


# Main training loop

train_dataset = ADNIBaselineDataset(csv_path, root_dir, transform, noisy_label_map)
test_dataset = ADNIBaselineDataset(test_csv, root_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in train_loader:
        inputs = batch["img"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    # Added evaluation per epoch
    test_metrics = evaluate(model, test_loader, device, loss_function)
    test_acc = test_metrics["accuracy"] * 100
    print(f"Epoch [{epoch+1}/{epochs}] | Test Accuracy: {test_acc:.2f}%")


# Final Evaluation
test_metrics, test_probs = evaluate(model, test_loader, device, loss_function, return_probs=True)
test_metrics_to_log = {
    "test/acc": test_metrics["accuracy"],
    "test/bacc": test_metrics["balanced_accuracy"],
    "test/f1_mac": test_metrics["f1_macro"],
    "test/f1_mic": test_metrics["f1_micro"],
    "test/auc": test_metrics["roc_auc"],
    "test/loss": test_metrics["loss"],
}

print("\nTest Results:")
for k, v in test_metrics_to_log.items():
    print(f"{k}: {v:.4f}")


# Save predictions with probabilities (for CI calculation)
baseline_preds = np.argmax(test_probs, axis=1)
baseline_ids = [sample["id"] for sample in test_dataset]
baseline_labels = [sample["label"] for sample in test_dataset]

# Create column names "Prob_Class_0", "Prob_Class_1"
prob_cols = {f"Prob_Class_{i}": test_probs[:, i] for i in range(test_probs.shape[1])}

# Combine into a DataFrame
df_out = pd.DataFrame({
    "PTID": baseline_ids,
    "True_Label": baseline_labels,
    "pred_class": baseline_preds,
    **prob_cols
})

output_path = ""
df_out.to_csv(output_path, index=False)
print(f"Saved probability output to {output_path}")

