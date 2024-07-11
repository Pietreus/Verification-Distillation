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
    def __init__(self, csv_file, target_size=None, noise_radius=1, transform=to_tensor, apply_noise=False,
                 labelling_model=None):
        data = pd.read_csv(csv_file)
        transform = transform

        # Separate features and labels
        self.features = torch.Tensor(data.iloc[:, :-1].values.astype(np.float32))
        self.labels = data.iloc[:, -1].values.astype(np.int64)

        # One-hot encode the labels
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.labels = torch.Tensor(one_hot_encoder.fit_transform(self.labels.reshape(-1, 1)))
        print(f"Number of data points: {self.features.shape[0]}")

        if apply_noise:
            # Calculate number of replicas needed
            original_size = self.features.shape[0]
            num_replicas = int(np.ceil(target_size / original_size))

            # Replicate data with noise
            self.features = np.vstack(
                [add_uniform_ball_noise(self.features.numpy(), noise_radius) for _ in range(num_replicas)])
            self.labels = np.vstack([self.labels.numpy() for _ in range(num_replicas)])
            print(f"Number of data points after noise: {self.features.shape[0]}")

            # Exclude original data
            self.features = torch.tensor(self.features[original_size:, :])
            self.labels = torch.tensor(self.labels[original_size:, :])

        # Apply transformations if provided
        if transform:
            self.features = transform(self.features)

        if labelling_model:
            outputs = labelling_model(torch.Tensor(self.features))
            self.labels = outputs  # torch.eye(outputs.max())[torch.argmax(outputs, dim=1)]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        return self.features[idx], self.labels[idx]
        # sample = self.features[idx]
        # label = self.labels[idx]
        # return sample, label


class SyntheticDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        self.features = data.clone().detach()
        self.labels = labels.clone().detach()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ToTensor:
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)
