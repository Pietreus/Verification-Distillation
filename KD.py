import io
import os
from datetime import datetime
import socket

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import SyntheticDataset
from RobustMockTeacher import MockNeuralNetwork

from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
# Get the hostname: e.g., thinkpad
hostname = socket.gethostname()
# Combine to form the full log directory path: e.g., my_subdir/May26_13-07-26_thinkpad
log_dir = os.path.join("runs", f"{current_time}_{hostname}")


def LGAD(x, y_true, y_pred_S, y_pred_T, lambda_CE=1.0, lambda_KL=4.0, lambda_GAD=1.0,
         temperature=1.0):
    LCE = torch.nn.CrossEntropyLoss()
    LKL = torch.nn.KLDivLoss(log_target=True)
    CE_loss = LCE(torch.nn.functional.softmax(y_pred_S, dim=1), y_true)
    KL_loss = temperature ** 2 * LKL(torch.nn.functional.log_softmax(y_pred_S, dim=1),
                                     torch.nn.functional.log_softmax(y_pred_T, dim=1))

    CE_loss_T = LCE(torch.nn.functional.softmax(y_pred_T, dim=1), y_true)
    # CE_loss_T.backward(retain_graph=True)
    teacher_grad = torch.autograd.grad(CE_loss_T, x, retain_graph=True, create_graph=True)[0]
    # CE_loss.backward(retain_graph=True)
    student_grad = torch.autograd.grad(CE_loss, x, retain_graph=True, create_graph=True)[0]
    grad_discrepancy = torch.norm(teacher_grad - student_grad)
    perc_grad_discrepancy = grad_discrepancy / torch.norm(student_grad)
    # grad_discrepancy = torch.norm(x.grad)
    # print(f"CE:{CE_loss},KL:{KL_loss},GAD:{grad_discrepancy}")

    return lambda_CE * CE_loss + lambda_KL * KL_loss + lambda_GAD * grad_discrepancy, CE_loss, KL_loss, grad_discrepancy, perc_grad_discrepancy


def save_tensors_for_later(teacher_outputs, student_outputs, synthetic_data, synthetic_labels,
                           high_confidence_teacher_mask, high_confidence_student_mask, log_dir, step):
    # Save tensors to disk
    os.makedirs(log_dir, exist_ok=True)
    torch.save(teacher_outputs, os.path.join(log_dir, f'teacher_outputs_{step}.pt'))
    torch.save(student_outputs, os.path.join(log_dir, f'student_outputs_{step}.pt'))
    torch.save(synthetic_data, os.path.join(log_dir, f'synthetic_data_{step}.pt'))
    torch.save(synthetic_labels, os.path.join(log_dir, f'synthetic_labels_{step}.pt'))
    torch.save(high_confidence_teacher_mask, os.path.join(log_dir, f'high_confidence_teacher_mask_{step}.pt'))
    torch.save(high_confidence_student_mask, os.path.join(log_dir, f'high_confidence_student_mask_{step}.pt'))


def log_networks_to_tensorboard(writer, teacher_model, student_model, synthetic_data, device, step, confidence=0.5,
                                **kwargs):
    # Ensure the models are in evaluation mode
    teacher_model.eval()
    student_model.eval()

    synthetic_data = synthetic_data.to(device)

    with torch.no_grad():
        teacher_outputs = torch.softmax(teacher_model(synthetic_data), dim=1)
        student_outputs = torch.softmax(student_model(synthetic_data), dim=1)

    synthetic_labels = torch.eye(2).to(device)[torch.argmax(teacher_outputs, dim=1)]
    high_confidence_teacher_mask = (torch.abs(teacher_outputs[:, 0] - 0.5) * 2 > confidence)
    high_confidence_student_mask = (torch.abs(student_outputs[:, 0] - 0.5) * 2 > confidence)

    # Move tensors to CPU for saving
    teacher_outputs = teacher_outputs.cpu()
    student_outputs = student_outputs.cpu()
    synthetic_data = synthetic_data.cpu()
    synthetic_labels = synthetic_labels.cpu()
    high_confidence_teacher_mask = high_confidence_teacher_mask.cpu()
    high_confidence_student_mask = high_confidence_student_mask.cpu()

    # Log additional configuration details as scalars
    for key, value in kwargs.items():
        writer.add_scalar(f'Config/{key}', value, step)

    # Save tensors for later processing
    save_tensors_for_later(teacher_outputs, student_outputs, synthetic_data, synthetic_labels,
                           high_confidence_teacher_mask, high_confidence_student_mask, writer.log_dir, step)


def plot_networks(teacher_model, student_model, synthetic_data, device, save=True, show=True,
                  l_GAD=1, l_CE=1, l_KD=1, confidence=0.5):
    # Ensure the models are in evaluation mode
    teacher_model.eval()
    student_model.eval()
    teacher_predictions = teacher_model(synthetic_data)
    synthetic_labels = torch.eye(2).to(device)[torch.argmax(teacher_predictions, dim=1)]

    # Convert synthetic data to a PyTorch tensor if it's not already
    if not isinstance(synthetic_data, torch.Tensor):
        synthetic_data = torch.tensor(synthetic_data, dtype=torch.float32)

    # Forward pass through the models to get the outputs
    with torch.no_grad():
        teacher_outputs = teacher_model(synthetic_data)
        student_outputs = student_model(synthetic_data)

    # Convert outputs to numpy arrays for plotting
    teacher_outputs = torch.softmax(teacher_outputs, dim=1).to("cpu").numpy()
    student_outputs = torch.softmax(student_outputs, dim=1).to("cpu").numpy()

    # Plotting
    fig, axes = plt.subplots(4, 2, figsize=(12, 18))

    # Teacher model output plots
    axes[0, 0].scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=teacher_outputs[:, 0], cmap='viridis', s=5)
    axes[0, 0].set_title(f'Teacher Model Output 1, GAD: {l_GAD}, CE: {l_CE}, KD: {l_KD}')
    axes[0, 1].scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=teacher_outputs[:, 1], cmap='viridis', s=5)
    axes[0, 1].set_title('Teacher Model Output 2')

    # Student model output plots
    axes[1, 0].scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=student_outputs[:, 0], cmap='viridis', s=5)
    axes[1, 0].set_title('Student Model Output 1')
    axes[1, 1].scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=student_outputs[:, 1], cmap='viridis', s=5)
    axes[1, 1].set_title('Student Model Output 2')

    # Synthetic labels plot
    axes[2, 0].scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=synthetic_labels[:, 0] > 0, cmap='viridis', s=5)
    axes[2, 0].set_title('Synthetic Labels')
    axes[2, 1].scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=student_outputs[:, 0] > 0.5, cmap='viridis', s=5)
    axes[2, 1].set_title('Student Labels')

    # Synthetic labels plot
    mask = np.abs(teacher_outputs[:, 0] - 0.5) * 2 > confidence
    axes[3, 0].scatter(synthetic_data[mask, 0], synthetic_data[mask, 1], c=synthetic_labels[mask, 0] > 0,
                       cmap='viridis', s=5)
    axes[3, 0].set_title('Synthetic Labels')
    mask = np.abs(student_outputs[:, 0] - 0.5) * 2 > confidence
    axes[3, 1].scatter(synthetic_data[mask, 0], synthetic_data[mask, 1], c=student_outputs[mask, 0] > 0.5,
                       cmap='viridis', s=5)
    axes[3, 1].set_title('Student High-confidence Labels')

    # axes[2, 1].axis('off')  # Turn off the last subplot as it's not needed

    # Adjust layout
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        # Convert buffer to an image and log to TensorBoard
        image = Image.open(buf)
        image = torch.tensor(np.array(image)).permute(2, 0, 1)  # Convert to CHW format
        writer.add_image('Model Outputs', image, 0)


def high_confidence_data(synthetic_data, model, confidence):
    mask = torch.flatten(torch.abs(model(synthetic_data)[:, 0]) > confidence, 0)
    return synthetic_data[mask]


def knowledge_distillation(distillation_data: torch.Tensor, teacher_model, student_model, batch_size, epochs, l_CE,
                           l_KD, l_GAD,
                           print_functions=False, device="cpu", confidence=0.5, teacher_freq=0.5):
    print(teacher_freq)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"confidence_{confidence}_teacher_freq_{teacher_freq}"))

    full_distillation_data = distillation_data
    # distillation_data = high_confidence_data(distillation_data,teacher_model, confidence)
    # Get teacher predictions for synthetic data
    teacher_predictions = teacher_model(distillation_data)

    # Convert teacher predictions to labels
    synthetic_labels = torch.eye(2).to(device)[torch.argmax(teacher_predictions, dim=1)]
    # synthetic_labels.to(device)

    # Train student model using synthetic data and teacher predictions
    optimizer = optim.Adam(student_model.parameters(), lr=0.0005)

    dataset = SyntheticDataset(distillation_data, synthetic_labels)

    # Create DataLoader
    for epoch in tqdm(range(epochs), ncols=50 + epochs):
    # for epoch in range(epochs):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            inputs.requires_grad = True

            # Forward pass
            outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs)
            # Set requires_grad=True on inputs to enable gradient computation
            # np.exp(-epoch / 100)
            # Compute your custom loss
            loss, ce, kl, gad, grad_ratio = LGAD(inputs, targets, outputs, teacher_outputs,
                                                 temperature=np.exp(-epoch / 100), lambda_GAD=l_GAD,
                                                 lambda_CE=l_CE, lambda_KL=l_KD)
            writer.add_scalar('Loss/Cross_Entropy', ce.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/KL_Divergence', kl.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/GradientDisparity', gad.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/Total', loss.item(), epoch * len(train_loader) + batch_idx)

            loss.backward()
            optimizer.step()
    writer.flush()
    writer.close()
    if print_functions:
        log_networks_to_tensorboard(writer, teacher_model, student_model, full_distillation_data, device, step=epochs,
                                    l_GAD=l_GAD, l_CE=l_CE, l_KD=l_KD,
                                    confidence=confidence)

    return loss, grad_ratio


if __name__ == "__main__":
    teacher = MockNeuralNetwork(5, 1)
    knowledge_distillation(teacher, teacher, 10 ** 6, (5,), 1000, 100)
