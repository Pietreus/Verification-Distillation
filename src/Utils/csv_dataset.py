import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def to_tensor(data): return torch.tensor(data, dtype=torch.float32)

def add_uniform_ball_noise(data, radius):
    n_samples, n_features = data.shape
    noise = np.random.uniform(-radius, radius, size=(n_samples, n_features))
    noise = noise / np.linalg.norm(noise, axis=1, keepdims=True) * np.random.uniform(0, radius, size=(n_samples, 1))
    return data + noise

class CSVDataset(Dataset):
    def __init__(self, csv_file, target_size=None, noise_radius=1, transform=to_tensor, apply_noise=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Separate features and labels
        self.features = self.data.iloc[:, :-1].values.astype(np.float32)
        self.labels = self.data.iloc[:, -1].values.astype(np.int64)

        # One-hot encode the labels
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.labels = self.one_hot_encoder.fit_transform(self.labels.reshape(-1, 1))

        print(f"Number of data points: {self.features.shape[0]}")

        if apply_noise:
            # Calculate number of replicas needed
            original_size = self.features.shape[0]
            num_replicas = int(np.ceil(target_size / original_size))

            # Replicate data with noise
            self.features = np.vstack(
                [add_uniform_ball_noise(self.features, noise_radius) for _ in range(num_replicas)])
            self.labels = np.vstack([self.labels for _ in range(num_replicas)])
            print(f"Number of data points after noise: {self.features.shape[0]}")

            # Exclude original data
            self.features = self.features[original_size:, :]
            self.labels = self.labels[original_size:, :]

        # Apply transformations if provided
        if self.transform:
            self.features = self.transform(self.features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.features[idx]
        label = self.labels[idx]
        return sample, label



class ToTensor:
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)
