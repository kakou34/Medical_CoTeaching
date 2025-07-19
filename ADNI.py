
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImage, Resize, NormalizeIntensity


class ADNI(Dataset):
    """
    ADNI dataset loader for 3D medical images.

    Args:
        root (string): Root directory where images are stored.
        csv_path (string): Path to CSV file with 'image_path' and 'binaryLabels' columns.
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
        

        self.data = df['PTID'].tolist()
        self.labels = df['binaryLabel'].tolist()

        print(f"train: {self.train}")
        print(f"noise type: {self.noise_type}")

        # Add label noise if specified
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
            
            filename = f"noise_maps/noise_map_{self.noise_type}_adni_{int(self.noise_rate * 100)}_seed{self.random_state}.csv"
            noiselist.to_csv(filename, index=False)
            print(f"[ADNI] Noise map saved to {filename}")

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

        unique_classes = np.unique(labels)
        num_classes = len(unique_classes)
        num_samples = len(labels)

        print(f"Random Seed: {self.random_state}, Noise Rate: {self.noise_rate}, Num Classes: {num_classes}")

        # Identity matrix (default: no noise)
        P = np.eye(num_classes)

        if self.noise_type == 'pairflip':
            if num_classes > 2:  # Multi-class
                for i in range(num_classes - 1):
                    P[i, i] = 1. - self.noise_rate
                    P[i, i + 1] = self.noise_rate
                P[num_classes - 1, num_classes - 1] = 1. - self.noise_rate
                P[num_classes - 1, 0] = self.noise_rate  # Cyclic pairflip
            else:  # Binary classification (0 ↔ 1)
                P = np.array([[1 - self.noise_rate, self.noise_rate],
                              [self.noise_rate, 1 - self.noise_rate]])

        elif self.noise_type == 'symmetric':
            P = np.ones((num_classes, num_classes)) * (self.noise_rate / (num_classes - 1))
            np.fill_diagonal(P, 1. - self.noise_rate)

        # Ensure P is row-stochastic
        assert np.allclose(P.sum(axis=1), 1), "Transition matrix rows do not sum to 1!"

        # Apply noise using the transition matrix
        rng = np.random.RandomState(self.random_state)

        for i in range(num_samples):
            current_label = labels[i]
            if current_label in unique_classes:
                label_index = np.where(unique_classes == current_label)[0][0]
                noisy_label_index = rng.choice(num_classes, p=P[label_index])
                noisy_labels[i] = unique_classes[noisy_label_index]

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
        img_path = os.path.join(self.root, f"{self.data[index]}.nii.gz")
        label = self.noisy_labels[index] if self.train else self.labels[index]

        # Load and transform the image
        img = img_path
        if self.transform is not None:
            img = self.transform(img)

        return img, label, index

    def __len__(self):
        return len(self.data)


# Define dataset transformations
def get_transforms(train=True):
    """
    Defines the transformations for the dataset.

    Args:
        train (bool): Whether to return training or testing transformations.

    Returns:
        Compose: MONAI Compose object with transformations.
    """
    return Compose([
        LoadImage(ensure_channel_first=True),
        Resize(spatial_size=(96, 116, 96)),
        NormalizeIntensity(),
    ])


if __name__ == "__main__":
    # Paths
    root_dir = "/projects/0/prjs0830/DATA/ADNI/Imaging/Preprocessed"
    train_csv = "/projects/0/prjs1245/Co-teaching/train.csv"
    test_csv = "/projects/0/prjs1245/Co-teaching/test_total.csv"

    # Train dataset and loader
    train_dataset = ADNI(
        root=root_dir,
        csv_path=train_csv,
        transform=get_transforms(train=True),
        noise_type='pairflip',  # Change to 'symmetric' if needed
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

    