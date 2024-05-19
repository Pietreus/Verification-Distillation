import numpy as np
# import tensorflow as tf
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import SyntheticDataset
from RobustMockTeacher import MockNeuralNetwork

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def LGAD(x, y_true, y_pred_S, y_pred_T, lambda_CE=1.0, lambda_KL=1.0, lambda_GAD=1.0,
         temperature=1.0):
    LCE = torch.nn.CrossEntropyLoss()
    LKL = torch.nn.KLDivLoss(log_target=True)
    CE_loss = LCE(y_pred_S, y_true)
    KL_loss = temperature ** 2 * LKL(torch.nn.functional.log_softmax(y_pred_S, dim=1),
                                     torch.nn.functional.log_softmax(y_pred_T, dim=1))

    CE_loss_T = -LCE(y_pred_T, y_true)
    CE_loss_T.backward(retain_graph=True)
    CE_loss.backward(retain_graph=True)
    grad_discrepancy = torch.norm(x.grad)
    # print(f"CE:{CE_loss},KL:{KL_loss},GAD:{grad_discrepancy}")

    return lambda_CE * CE_loss + lambda_KL * KL_loss + lambda_GAD * grad_discrepancy, CE_loss, KL_loss, grad_discrepancy


def plot_networks(teacher_model, student_model, synthetic_data, synthetic_labels):
    # Ensure the models are in evaluation mode
    teacher_model.eval()
    student_model.eval()

    # Convert synthetic data to a PyTorch tensor if it's not already
    if not isinstance(synthetic_data, torch.Tensor):
        synthetic_data = torch.tensor(synthetic_data, dtype=torch.float32)

    # Forward pass through the models to get the outputs
    with torch.no_grad():
        teacher_outputs = teacher_model(synthetic_data)
        student_outputs = student_model(synthetic_data)

    # Convert outputs to numpy arrays for plotting
    teacher_outputs = teacher_outputs.numpy()
    student_outputs = student_outputs.numpy()

    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))

    # Teacher model output plots
    axes[0, 0].scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=teacher_outputs[:, 0], cmap='viridis', s=5)
    axes[0, 0].set_title('Teacher Model Output 1')
    axes[0, 1].scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=teacher_outputs[:, 1], cmap='viridis', s=5)
    axes[0, 1].set_title('Teacher Model Output 2')

    # Student model output plots
    axes[1, 0].scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=student_outputs[:, 0], cmap='viridis', s=5)
    axes[1, 0].set_title('Student Model Output 1')
    axes[1, 1].scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=student_outputs[:, 1], cmap='viridis', s=5)
    axes[1, 1].set_title('Student Model Output 2')

    # Synthetic labels plot
    axes[2, 0].scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=synthetic_labels[:, 0]>0, cmap='viridis', s=5)
    axes[2, 0].set_title('Synthetic Labels')
    axes[2, 1].scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=student_outputs[:, 0]>0, cmap='viridis', s=5)
    axes[2, 1].set_title('Student Labels')

    # axes[2, 1].axis('off')  # Turn off the last subplot as it's not needed

    # Adjust layout
    plt.tight_layout()
    plt.show()


def knowledge_distillation(teacher_model, student_model, num_samples, input_shape, batch_size, epochs, print_functions=False):

    # Generate synthetic data using normal distribution
    synthetic_data = torch.tensor(np.random.uniform(-2,2,size=(num_samples, *input_shape)), dtype=torch.float32)

    # Get teacher predictions for synthetic data
    teacher_predictions = teacher_model(synthetic_data)

    # Convert teacher predictions to labels
    synthetic_labels = torch.eye(2)[torch.argmax(teacher_predictions, dim=1)]

    # Train student model using synthetic data and teacher predictions
    optimizer = optim.Adam(student_model.parameters(), lr=0.0005)

    dataset = SyntheticDataset(synthetic_data, synthetic_labels)

    # Create DataLoader
    for epoch in tqdm(range(epochs)):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            inputs.requires_grad = True

            # Forward pass
            outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs)
            # Set requires_grad=True on inputs to enable gradient computation

            # Compute your custom loss
            loss, ce, kl, gad = LGAD(inputs, targets, outputs, teacher_outputs, temperature=np.exp(-epoch / 20))
            writer.add_scalar('Loss/Cross_Entropy', ce.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/KL_Divergence', kl.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/GradientDisparity', gad.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/Total', loss.item(), epoch * len(train_loader) + batch_idx)

            loss.backward()
            optimizer.step()
    writer.flush()
    writer.close()
    if print_functions:
        plot_networks(teacher_model,student_model,synthetic_data,synthetic_labels)


if __name__ == "__main__":
    teacher = MockNeuralNetwork(5, 1)
    knowledge_distillation(teacher, teacher, 10 ** 6, (5,), 1000, 100)
