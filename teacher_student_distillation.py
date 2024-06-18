import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tqdm import tqdm

from KD import LGAD
from torch.utils.tensorboard import SummaryWriter

from nnsaver import nnet_exporter

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


def add_uniform_ball_noise(data, radius):
    n_samples, n_features = data.shape
    noise = np.random.uniform(-radius, radius, size=(n_samples, n_features))
    noise = noise / np.linalg.norm(noise, axis=1, keepdims=True) * np.random.uniform(0, radius, size=(n_samples, 1))
    return data + noise


class CSVDataset(Dataset):
    def __init__(self, csv_file, target_size=None, noise_radius=1, transform=None, apply_noise=False,
                 one_hot_encode=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Separate features and labels
        self.features = self.data.iloc[:, :-1].values.astype(np.float32)
        self.labels = self.data.iloc[:, -1].values.astype(np.int64)

        if one_hot_encode:
            self.one_hot_encoder = OneHotEncoder(sparse_output=False)
            self.labels = self.one_hot_encoder.fit_transform(self.labels.reshape(-1, 1))

        print(f"Number of data points: {self.features.shape[0]}\n")

        # Replicates the data to match target_size, and applies noise to each replica.
        if apply_noise:
            original_size = self.features.shape[0]
            num_replicas = int(np.ceil(target_size / original_size))
            self.features = np.vstack(
                [add_uniform_ball_noise(self.features, noise_radius) for _ in range(num_replicas)])
            self.labels = np.vstack([self.labels for _ in range(num_replicas)])
            print(f"Number of data points after noise: {self.features.shape[0]}")

            # Exclude original data
            self.features = self.features[original_size:, :]
            self.labels = self.labels[original_size:, :]

        if self.transform:
            self.features = self.transform(self.features)

    def __len__(self):
        # return len(self.data)
        return self.features.shape[0]

    def __getitem__(self, idx):
        sample = self.features[idx]
        label = self.labels[idx]
        return sample, label


class ToTensor:
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)


def disagreement_check(dataset, teacher, student):

    data_loader = DataLoader(dataset, batch_size=128)
    optimizer = optim.Adam(student.parameters(), lr=0.0005)

    differences = []
    grad_discrepancies = []
    student_grads = []
    teacher_grads = []
    ratios = []
    difference_ratios = []
    median_directional = []
    median_directional_all = []
    min_directional = []
    mean_diff_directional = []
    max_diff_directional = []

    for samples, labels in data_loader:
        optimizer.zero_grad()

        samples.requires_grad = True
        student_outputs = student(samples)
        teacher_outputs = teacher(samples)
        student_predicted = torch.max(student_outputs.data, 1)[0]
        teacher_predicted = torch.max(teacher_outputs.data, 1)[0]
        y_true = torch.max(labels, 1)[1]

        # Max point value disagreement.
        difference = torch.max(
            torch.abs(torch.max(student_outputs.data, 1)[0] - torch.max(teacher_outputs.data, 1)[0]))
        differences.append(difference)
        difference_ratios.append((student_outputs.data - teacher_outputs.data).max() / student_outputs.data.max())

        # Computing CE gradient
        LCE = torch.nn.CrossEntropyLoss()
        # CE_loss_T = LCE(torch.nn.functional.softmax(student_outputs, dim=1), labels)
        # CE_loss_S = LCE(torch.nn.functional.softmax(teacher_outputs, dim=1), labels)
        CE_loss_T = LCE(student_outputs, labels)
        CE_loss_S = LCE(teacher_outputs, labels)

        teacher_grad = torch.autograd.grad(CE_loss_T, samples, retain_graph=True, create_graph=True)[0]
        student_grad = torch.autograd.grad(CE_loss_S, samples, retain_graph=True, create_graph=True)[0]

        teacher_grads.append(torch.min(torch.abs(teacher_grad)))
        student_grads.append(torch.norm(student_grad))
        ratios.append(torch.norm(student_grad) / torch.norm(teacher_grad))

        # Direction_wise disagreement
        ratio = torch.abs(student_grad/teacher_grad)
        bad_ratios = ratio[ratio < 1]

        median_directional.append(bad_ratios.median())
        median_directional_all.append(ratio.median())
        min_directional.append(bad_ratios.min())
        mean_diff_directional.append(torch.mean(teacher_grad - student_grad))
        max_diff_directional.append(torch.max(teacher_grad - student_grad))

        grad_discrepancy = torch.norm(teacher_grad - student_grad)
        grad_discrepancies.append(grad_discrepancy)

    # print(torch.max(torch.stack(differences)))  # Function values
    # print(torch.max(torch.stack(grad_discrepancies)))  # Gradient abs difference
    print(torch.min(torch.stack(teacher_grads)))
    # print(torch.max(torch.stack(student_grads)))
    # print(torch.max(torch.stack(ratios)))  # student_grad/teacher_grad
    # print(torch.max(torch.stack(difference_ratios)))  # ratio of difference in prediction
    print(f"Median ratio directional, for < 1: {torch.median(torch.stack(median_directional))}")
    print(f"Median ratio directional, for all: {torch.median(torch.stack(median_directional_all))}")
    print(f"Min ratio directional, for < 1: {torch.min(torch.stack(min_directional))}")
    print(f"Mean difference directional disagreement: {torch.mean(torch.stack(mean_diff_directional))}")
    print(f"Max difference directional disagreement: {torch.max(torch.stack(max_diff_directional))}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    ############################
    # Train teacher.
    ############################
    csv_file = 'datasets/compas-scores-preprocessed.csv'
    dataset = CSVDataset(csv_file, transform=ToTensor(), target_size=100, apply_noise=False)

    # Train and validation sets.
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize.
    input_dim = dataset.features.shape[1]
    num_classes = 3
    teacher = TeacherModel(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
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

    # Validation.
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

    ############################
    # Distill student.
    ############################

    student = StudentModel(input_dim, num_classes)
    optimizer = optim.Adam(student.parameters(), lr=0.0005)

    # Knowledge distillation.
    print("Performing Knowledge Distillation")
    noise_radius = 1
    distillation_data = CSVDataset(csv_file, transform=ToTensor(), target_size=10e4,
                                   apply_noise=True, noise_radius=noise_radius)

    l_GAD = 50
    l_CE = 2
    l_KD = 5

    epochs = 50

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
                                               lambda_GAD=l_GAD, lambda_CE=l_CE, lambda_KL=l_KD, softmax=False)
            writer.add_scalar('Loss/Cross_Entropy', ce.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/KL_Divergence', kl.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/GradientDisparity', gad.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/GrPercDisparity', gad_perc.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/Total', loss.item(), epoch * len(train_loader) + batch_idx)

            loss.backward()
            optimizer.step()

        # disagreement_check(distillation_data, teacher=teacher, student=student)
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

    nnet_exporter(student, f"models/nnet/student_noise_{noise_radius}_CE_{l_CE}_KL_{l_KD}_GAD_{l_GAD}.nnet", dataset)
    nnet_exporter(teacher, f"models/nnet/teacher_noise_{noise_radius}_CE_{l_CE}_KL_{l_KD}_GAD_{l_GAD}.nnet", dataset)

    # Checking for disagreement.
    disagreement_check(distillation_data, teacher=teacher, student=student)
