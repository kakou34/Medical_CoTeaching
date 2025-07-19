import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImage, NormalizeIntensity, RandRotate, RandGaussianNoise, RandAdjustContrast

class OrganMNIST3D(Dataset):
    """
    OrganMNIST3D dataset loader for 3D medical images.

    Args:
        root (string): Root directory where images are stored.
        csv_path (string): Path to CSV file with 'image_path' and 'organ_label' columns.
        transform (callable, optional): Transformations to apply to the 3D images.
        noise_type (str, optional): Type of label noise to add (e.g., 'pairflip', 'symmetric').
        noise_rate (float, optional): Corruption rate for noisy labels.
    """
    def __init__(self, root, csv_path, transform=None, noise_type=None, noise_rate=0.0, random_state=0, train=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.random_state = random_state
        self.train = train

        # Load data and labels from CSV
        df = pd.read_csv(csv_path)
        self.data = df['img'].tolist()
        self.labels = df['label'].tolist() 

        print(f"train: {self.train}")
        print(f"noise type: {self.noise_type}")

        # Add label noise if specified
        if self.train and self.noise_type is not None and self.noise_type != 'clean':
            self.labels = np.array(self.labels)
            self.noisy_labels, self.actual_noise_rate = self._add_noise(self.labels)
        else:
            self.noisy_labels = self.labels
            print("Labels not changed")

        # Determine which samples have noisy labels
        self.noise_or_not = (np.array(self.labels) == np.array(self.noisy_labels)).astype(int).tolist()

        if self.train and self.noise_type is not None and self.noise_type != 'clean':
            os.makedirs("noise_maps", exist_ok=True)
            noiselist = pd.DataFrame({
                'Sample_ID': self.data,
                'Original_Label': self.labels,
                'Noisy_Label': self.noisy_labels,
                'Noise_Status': self.noise_or_not
        })
            
            filename = f"noise_maps/noise_map_{self.noise_type}_{int(self.noise_rate * 100)}_seed{self.random_state}.csv"
            noiselist.to_csv(filename, index=False)
            print(f"[OrganMNIST3D] Noise map saved to {filename}")

    def _add_noise(self, labels):
        """
        Adds structured noise using a transition probability matrix.

        Returns:
            noisy_labels (array): Labels after noise injection.
            actual_noise_rate (float): Measured noise rate.
        """
        print("Debug: Entering _add_noise...")

        np.random.seed(self.random_state)
        labels = np.array(labels)  
        noisy_labels = labels.copy()

        unique_classes = np.unique(labels)  # Get sorted unique class labels
        num_classes = len(unique_classes)  # Total number of classes
        num_samples = len(labels)

        print(f"Random Seed: {self.random_state}, Noise Rate: {self.noise_rate}, Num Classes: {num_classes}")

        # Identity matrix (no noise by default)
        P = np.eye(num_classes)

        if self.noise_type == 'pairflip':
            for i in range(num_classes - 1):  
                P[i, i] = 1. - self.noise_rate  # Stay the same
                P[i, i + 1] = self.noise_rate   # Flip to next class
            P[num_classes - 1, num_classes - 1] = 1. - self.noise_rate
            P[num_classes - 1, 0] = self.noise_rate  # Last class flips to first (cyclic pairflip)

        elif self.noise_type == 'symmetric':
            P = np.ones((num_classes, num_classes)) * (self.noise_rate / (num_classes - 1))
            np.fill_diagonal(P, 1. - self.noise_rate)  # Keep original label with probability (1 - noise_rate)

        # Ensure P is row-stochastic (each row sums to 1)
        assert np.allclose(P.sum(axis=1), 1), "Transition matrix rows do not sum to 1!"

        # Apply noise using the transition matrix
        rng = np.random.RandomState(self.random_state)

        for i in range(num_samples):
            current_label = labels[i]
            if current_label in unique_classes:
                label_index = np.where(unique_classes == current_label)[0][0]  # Find index in unique classes
                noisy_label_index = rng.choice(num_classes, p=P[label_index])  # Sample new label
                noisy_labels[i] = unique_classes[noisy_label_index]  # Map back to original class values

        actual_noise_rate = (noisy_labels != labels).mean()

        print(f"Actual Noise Rate: {actual_noise_rate:.2f}")

        return noisy_labels, actual_noise_rate

    def __getitem__(self, index):
        """
        Fetches a single sample.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            tuple: (image, label, index)
        """
        # Construct full image path from 'root' and 'img' column
        img_path = os.path.join(self.root, self.data[index]) 
        # Ensure file exists before proceeding
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # Apply MONAI transforms (expects a path string, not a tensor)
        if self.transform is not None:
            img = self.transform(img_path)

        # Get the corresponding label
        label = self.noisy_labels[index]

        # Return the image, label, and index
        return img, label, index

    def __len__(self):
        return len(self.data)
    
    
def get_transforms(train=True):
    """
    Defines the transformations for the dataset.

    Args:
        train (bool): Whether to return training or testing transformations.

    Returns:
        Compose: MONAI Compose object with transformations.
    """
    if train:
        return Compose([
            LoadImage(ensure_channel_first=True),
            NormalizeIntensity(nonzero=True, channel_wise=True),
            RandRotate(range_x=15, range_y=15, range_z=15, prob=0.5),
            RandGaussianNoise(prob=0.1),
            RandAdjustContrast(prob=0.2)
        ])
    else:
        return Compose([
            LoadImage(ensure_channel_first=True),
            NormalizeIntensity(nonzero=True, channel_wise=True),
        ])


if __name__ == "__main__":
      # Dataset Paths
    train_csv = "/projects/0/prjs1245/Co-teaching/organ_train.csv"
    test_csv = "/projects/0/prjs1245/Co-teaching/organ_test.csv"
    root_dir = "/projects/0/prjs0830/DATA/OrganMNIST/Imaging/Preprocessed"

    # Train dataset and loader
    train_dataset = OrganMNIST3D(
        root=root_dir,
        csv_path=train_csv,
        transform=get_transforms(train=True),
        noise_type='pairflip',  # Can be 'symmetric' or None
        noise_rate=0.2,
        train=True
    )
    print("Train dataset created with length:", len(train_dataset))

    test_dataset = OrganMNIST3D(
        root=root_dir,
        csv_path=test_csv,
        transform=get_transforms(train=False),
        train=False
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    for batch in train_loader:
        images, labels, indices = batch
        print(f"Train - Images shape: {images.shape}, Labels: {labels}, Indices: {indices}")

    for batch in test_loader:
        images, labels, indices = batch
        print(f"Test - Images shape: {images.shape}, Labels: {labels}, Indices: {indices}")


