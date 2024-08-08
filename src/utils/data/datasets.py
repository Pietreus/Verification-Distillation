import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from sklearn import datasets as sk_datasets
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


def get_loaders(dataset_name, batch_size=32, val_split=0.2, test_split=0.2, random_state=42, flatten=False):
    if dataset_name.lower() == 'iris':
        iris = sk_datasets.load_iris()
        X, y = iris.data, iris.target
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_split + val_split,
                                                            random_state=random_state)
        if val_split > 0.0:
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                            test_size=test_split / (test_split + val_split),
                                                            random_state=random_state)
        else:
            X_val = X_temp
            X_test = X_temp
            y_val = y_temp
            y_test = y_temp

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

    elif dataset_name.lower() == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if flatten:
            flatten_transform = transforms.Compose([
                transforms.Lambda(lambda x: x.view(-1))  # Flatten.
                ])
            transform = transforms.Compose([transform, flatten_transform])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_size = int((1 - val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    elif dataset_name.lower() == 'susy': #ignores the test split, as there is a designated test set
        full_dataset = torch.tensor(np.loadtxt("./datasets/SUSY.csv", dtype=float, delimiter=","))
        X, y = full_dataset[:4500000, :-1], full_dataset[4500000:, -1]
        X_test, y_test = full_dataset[4500000:, :-1], full_dataset[4500000:, -1]
        X_train, y_train, X_val, y_val = train_test_split(X, y, test_size=val_split,
                         random_state=random_state)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

    else:
        raise ValueError("Dataset not supported. Please choose 'iris' or 'mnist'.")

    # Dataloaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class NoisyDataset(Dataset):
    def __init__(self, original_dataset, noise_type='gaussian', noise_std=0.1, copies_per_sample=1):
        """
        Args:
            original_dataset (Dataset): The original dataset.
            noise_std (float): The standard deviation of the Gaussian noise.
            copies_per_sample (int): Number of noisy copies per original data point.
        """
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.copies_per_sample = copies_per_sample
        self.data, self.labels = self._create_noisy_data(original_dataset)

    def _create_noisy_data(self, original_dataset):
        noisy_data = []
        noisy_labels = []
        for data, label in original_dataset:
            for _ in range(self.copies_per_sample):
                noisy_data.append(self._add_noise(data))
                noisy_labels.append(label)

        noisy_data = torch.stack(noisy_data)
        noisy_labels = torch.tensor(noisy_labels)
        return noisy_data, noisy_labels

    def _add_noise(self, data):
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(data) * self.noise_std
        elif self.noise_type == 'uniform_ball':
            noise = self._generate_uniform_ball_noise(data.shape, self.noise_std)
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")

        return data + noise

    def _generate_uniform_ball_noise(self, shape, radius):
        """
        Generates uniform noise inside an n-dimensional ball of specified radius.
        """
        noise = torch.randn(shape)  # Sample from standard normal distribution
        norm = torch.norm(noise, dim=-1, keepdim=True)  # Compute norm
        uniform_noise = noise / norm  # Normalize to unit ball
        uniform_noise *= radius * torch.rand(shape[:-1], device=noise.device).unsqueeze(
            -1)  # Scale by uniform distribution over [0, radius]
        return uniform_noise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_noisy_loader(original_loader, noise_type='gaussian', noise_std=0.1, copies_per_sample=1):
    """
    Args:
        original_loader (DataLoader): The original DataLoader.
        noise_type (str): The type of noise to add ('gaussian' or 'uniform_ball').
        noise_std (float): The standard deviation for Gaussian noise or scale for uniform noise.
        copies_per_sample (int): Number of noisy copies per original data point.

    Returns:
        DataLoader: A new DataLoader with noisy data.
    """
    noisy_dataset = NoisyDataset(original_loader.dataset, noise_type, noise_std, copies_per_sample)
    noisy_loader = DataLoader(noisy_dataset, batch_size=original_loader.batch_size, shuffle=True)
    return noisy_loader


def get_input_output_dimensions(train_loader):
    inputs, labels = next(iter(train_loader))
    input_dim = inputs.shape[1:]  # Exclude batch size

    if labels.ndim == 1:
        output_dim = (len(labels.unique()),)
    else:
        output_dim = labels.shape[1:]

    return input_dim[0], output_dim
