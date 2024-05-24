import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tqdm import tqdm

from KD import LGAD
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


# TODO: to be put in a "model" class.
class TeacherModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class StudentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 6)
        self.fc2 = nn.Linear(6, 5)
        self.fc3 = nn.Linear(5, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)


    def add_uniform_ball_noise(data, radius):
        n_samples, n_features = data.shape
        noise = np.random.uniform(-radius, radius, size=(n_samples, n_features))
        noise = noise / np.linalg.norm(noise, axis=1, keepdims=True) * np.random.uniform(0, radius, size=(n_samples, 1))
        return data + noise

    # Train teacher.
    class CSVDataset(Dataset):
        def __init__(self, csv_file, target_size=None, noise_radius=1, transform=None, apply_noise=False):
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
                self.features = np.vstack([add_uniform_ball_noise(self.features, noise_radius) for _ in range(num_replicas)])
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


    # Define transformations (optional)
    class ToTensor:
        def __call__(self, sample):
            return torch.tensor(sample, dtype=torch.float32)


    # Load and preprocess dataset
    csv_file = 'datasets/compas-scores-preprocessed.csv'
    dataset = CSVDataset(csv_file, transform=ToTensor(), target_size=100, apply_noise=False)

    # Split into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    # Initialize model, loss function, and optimizer
    input_dim = dataset.features.shape[1]
    num_classes = 3
    teacher = TeacherModel(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5

    for epoch in range(num_epochs):
        teacher.train()
        running_loss = 0.0
        for samples, labels in train_loader:
            optimizer.zero_grad()

            outputs = teacher(samples)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Validation (optional)
    teacher.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for samples, labels in val_loader:
            outputs = teacher(samples)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()  # Convert one-hot to class indices

    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

    # Distill student.

    student = StudentModel(input_dim, num_classes)
    optimizer = optim.Adam(student.parameters(), lr=0.0005)

    # Knowledge distillation.
    print("Performing Knowledge Distillation")
    noise_radius = 1
    distillation_data = CSVDataset(csv_file, transform=ToTensor(), target_size=10e7,
                                   apply_noise=True, noise_radius=noise_radius)

    l_GAD = 50
    l_CE = 2
    l_KD = 5

    epochs = 100

    for epoch in tqdm(range(epochs)):

        distillation_loader = DataLoader(distillation_data, batch_size=128, shuffle=True)

        for batch_idx, (inputs, targets) in enumerate(distillation_loader):
            # Zero the parameter gradients.
            optimizer.zero_grad()
            inputs.requires_grad = True

            # Forward pass.
            outputs = student(inputs)
            teacher_outputs = teacher(inputs)

            loss, ce, kl, gad, gad_perc = LGAD(inputs, targets, outputs, teacher_outputs, temperature=1,
                                               lambda_GAD=l_GAD, lambda_CE=l_CE, lambda_KL=l_KD)
            writer.add_scalar('Loss/Cross_Entropy', ce.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/KL_Divergence', kl.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/GradientDisparity', gad.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/GrPercDisparity', gad_perc.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/Total', loss.item(), epoch * len(train_loader) + batch_idx)

            loss.backward()
            optimizer.step()
    writer.flush()
    writer.close()



    # Validation (optional)
    student.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for samples, labels in val_loader:
            outputs = student(samples)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()  # Convert one-hot to class indices

    print(f'Student Validation Accuracy: {100 * correct / total:.2f}%')

    # Saving models.
    torch.onnx.export(student, args=(val_dataset[0][0]),
                      f=f"models/student_noise_{noise_radius}_CE_{l_CE}_KL_{l_KD}_GAD_{l_GAD}.onnx")
    torch.save(student.state_dict(), f"models/student_noise_{noise_radius}_CE_{l_CE}_KL_{l_KD}_GAD_{l_GAD}.pt")

    torch.onnx.export(teacher, args=(val_dataset[0][0]),
                      f=f"models/teacher_noise_{noise_radius}_CE_{l_CE}_KL_{l_KD}_GAD_{l_GAD}.onnx")
    torch.save(teacher.state_dict(), f"models/teacher_noise_{noise_radius}_CE_{l_CE}_KL_{l_KD}_GAD_{l_GAD}.pt")



