import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        self.data = data.clone().detach()
        self.labels = labels.clone().detach()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
