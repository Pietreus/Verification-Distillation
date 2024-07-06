import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import Dataset, random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.Utils.Relu_network import FFNetwork
from src.Utils.csv_dataset import CSVDataset
from src.Utils.knowledge_distillation import knowledge_distillation_training

writer = SummaryWriter()


def print_disagreement_check(dataset, teacher, student):
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

        # Max point value disagreement.
        difference = torch.max(
            torch.abs(torch.max(student_outputs.data, 1)[0] - torch.max(teacher_outputs.data, 1)[0]))
        differences.append(difference)
        difference_ratios.append((student_outputs.data - teacher_outputs.data).max() / student_outputs.data.max())

        # Computing CE gradient
        LCE = torch.nn.CrossEntropyLoss()
        CE_loss_T = LCE(student_outputs, labels)
        CE_loss_S = LCE(teacher_outputs, labels)

        teacher_grad = torch.autograd.grad(CE_loss_T, samples, retain_graph=True, create_graph=True)[0]
        student_grad = torch.autograd.grad(CE_loss_S, samples, retain_graph=True, create_graph=True)[0]

        teacher_grads.append(torch.norm(teacher_grad))
        student_grads.append(torch.norm(student_grad))
        ratios.append(torch.norm(student_grad) / torch.norm(teacher_grad))

        # Direction_wise disagreement
        ratio = torch.abs(student_grad / teacher_grad)
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

    # Load and preprocess dataset
    csv_file = 'datasets/compas-scores-preprocessed.csv'
    dataset = CSVDataset(csv_file, target_size=100, apply_noise=False)

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
    teacher = FFNetwork(input_dim=input_dim, output_dim=num_classes, layer_sizes=[10, 10, 10, 10])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5

    ## TEACHER TRAINING ##
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

        print(f"teacher training: Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Validation (optional)
    teacher.eval()
    # Validation (optional)
    teacher.eval()
    with torch.no_grad():
        # for samples, labels in val_loader:
        outputs = teacher(val_dataset[:, 0])
        _, predicted = torch.max(outputs.data, 1)

        print(f'Student Validation Accuracy: {accuracy_score(val_dataset[:, 1], predicted)}%')

    # Distill student.

    student = FFNetwork(input_dim=input_dim, output_dim=num_classes, layer_sizes=[6, 5])
    optimizer = optim.Adam(student.parameters(), lr=0.0005)

    # Knowledge distillation.
    print("Performing Knowledge Distillation")
    noise_radius = 1
    distillation_data = CSVDataset(csv_file, target_size=10e7,
                                   apply_noise=True, noise_radius=noise_radius)

    l_GAD = 50
    l_CE = 2
    l_KD = 5
    epochs = 50

    knowledge_distillation_training(distillation_data, 3, teacher, student, l_GAD, l_CE, l_KD, epochs,
                                    log_writer=writer)
    writer.close()

    # Validation (optional)
    student.eval()
    with torch.no_grad():
        # for samples, labels in val_loader:
        outputs = student(val_dataset[:, 0])
        _, predicted = torch.max(outputs.data, 1)

        print(f'Student Validation Accuracy: {accuracy_score(val_dataset[:, 1], predicted)}%')

    # Saving models.
    torch.save(student.state_dict(), f"models/student_noise_{noise_radius}_CE_{l_CE}_KL_{l_KD}_GAD_{l_GAD}.pt")
    torch.save(teacher.state_dict(), f"models/teacher_noise_{noise_radius}_CE_{l_CE}_KL_{l_KD}_GAD_{l_GAD}.pt")

    # Checking for disagreement.

    print_disagreement_check(distillation_data, teacher=teacher, student=student)
