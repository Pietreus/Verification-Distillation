import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Assuming synthetic_data and synthetic_labels are already defined as numpy arrays
# synthetic_data = np.random.normal(size=(num_samples, *input_shape))
# synthetic_labels = np.argmax(teacher_predictions, axis=1)
# synthetic_labels = tf.keras.utils.to_categorical(synthetic_labels)

class SyntheticDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
