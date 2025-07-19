import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImage, Resize, NormalizeIntensity


class ADNI(Dataset):
    def __init__(self, root, csv_path, transform=None, noise_type=None, noise_rate=0.0, random_state=0, train=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.random_state = random_state
        self.train = train
        self.csv_path = csv_path  # <- Fix: store CSV path as instance variable

        df = pd.read_csv(csv_path)
        print("Available Columns:", df.columns.tolist())
        print(df.head())

        self.data = df['PTID'].tolist()
        self.labels = df['binaryLabel'].tolist()

        self.ages = df['AGE'].tolist() if 'AGE' in df.columns else None

        if self.train and self.noise_type is not None and self.noise_type != 'clean':
            self.labels = np.array(self.labels)
            self.noisy_labels, self.actual_noise_rate = self._add_noise(self.labels)
        else:
            self.noisy_labels = self.labels
            print("Labels not changed.")

        self.noise_or_not = (np.array(self.labels) == np.array(self.noisy_labels)).astype(int).tolist()

        if self.train and self.noise_type is not None and self.noise_type != 'clean':
            os.makedirs("noise_maps", exist_ok=True)
            noiselist = pd.DataFrame({
                'Sample_ID': self.data,
                'Original_Label': self.labels,
                'Noisy_Label': self.noisy_labels,
                'Noise_Status': self.noise_or_not
            })
            filename = f"noise_maps/noise_map_{self.noise_type}_adni__age{int(self.noise_rate * 100)}_seed{self.random_state}.csv"
            noiselist.to_csv(filename, index=False)
            print(f"[ADNI] Noise map saved to {filename}")

    def _add_noise(self, labels):
        print("Debug: Entering _add_noise...")

        np.random.seed(self.random_state)
        labels = np.array(labels)
        noisy_labels = labels.copy()

        df = pd.read_csv(self.csv_path)
        ages = df['AGE'].values

    
        indices_to_flip = np.where((ages > 70) & (labels == 0))[0]
        num_to_flip = int(self.noise_rate * len(indices_to_flip))
        flip_indices = np.random.choice(indices_to_flip, num_to_flip, replace=False)
        noisy_labels[flip_indices] = 1

        actual_noise_rate = (noisy_labels != labels).mean()

        print(f"Actual Noise Rate: {actual_noise_rate:.2f}")
        return noisy_labels, actual_noise_rate

    def __getitem__(self, index):
        img_path = os.path.join(self.root, f"{self.data[index]}.nii.gz")
        label = self.noisy_labels[index] if self.train else self.labels[index]
        img = img_path
        if self.transform is not None:
            img = self.transform(img)
        return img, label, index

    def __len__(self):
        return len(self.data)


def get_transforms(train=True):
    return Compose([
        LoadImage(ensure_channel_first=True),
        Resize(spatial_size=(96, 116, 96)),
        NormalizeIntensity(),
    ])


if __name__ == "__main__":
    root_dir = "/projects/0/prjs0830/p1_data/ADNI_pp"
    train_csv = "/projects/0/prjs1245/Co-teaching/train.csv"
    test_csv = "/projects/0/prjs1245/Co-teaching/test_total.csv"

    train_dataset = ADNI(
        root=root_dir,
        csv_path=train_csv,
        transform=get_transforms(train=True),
        noise_type='pairflip',
        noise_rate=0.2,
        train=True
    )
    print("Train dataset created with length:", len(train_dataset))

    test_dataset = ADNI(
        root=root_dir,
        csv_path=test_csv,
        transform=get_transforms(train=False),
        train=False
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    for batch in train_loader:
        images, labels, indices = batch
        print(f"Train - Images shape: {images.shape}, Labels: {labels}, Indices: {indices}")
        break

    for batch in test_loader:
        images, labels, indices = batch
        print(f"Test - Images shape: {images.shape}, Labels: {labels}, Indices: {indices}")
        break
