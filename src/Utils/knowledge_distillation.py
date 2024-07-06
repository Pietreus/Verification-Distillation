import numpy as np
import torch

from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def LGAD(inputs: torch.Tensor,
         true_labels: torch.Tensor,
         student_predictions: torch.Tensor,
         teacher_predictions: torch.Tensor,
         lambda_CE: float = 1.,
         lambda_KL: float = 1.,
         lambda_GAD: float = 1.,
         temperature: float = 1.) -> (torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module):
    """
    Gradient alignment distillation loss function.
    Takes as input a batch of n inputs, labels for c classes as well as
    the raw predictions of a teacher and student model.
    The loss consists of three terms:
    - students Categorical cross-entropy after application of softmax, weighted by lambda_CE
    - student-teacher KL divergence after application of softmax, weighted by lambda_KL and temperature
    - student-teacher gradient alignment with respect to the input after application of softmax, weighted by lambda_GAD
    TODO: link to paper
    :param inputs: (n,d) tensor, the input of the current batch
    :param true_labels: (n,c) tensor, binary labels for the correct output classes
    :param student_predictions: (n,c) tensor, raw model output of the student
    :param teacher_predictions: (n,c) tensor, raw model output of the teacher
    :param lambda_CE: >=0, weighting parameter for the categorical cross-entropy
    :param lambda_KL: >=0, weighting parameter for the KL divergence
    :param lambda_GAD: >=0, weighting parameter for the gradient alignment
    :param temperature: >=0, weighting parameter for the KL divergence
    :return (loss, CE_loss, KL_loss, GAD_loss, grad_diff, rel_grad_diff): a tuple containing the weighted loss
    as well as the three individual components and an additional ratio of gradients.
    """

    LCE = torch.nn.CrossEntropyLoss()
    LKL = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")
    CE_loss = LCE(torch.nn.functional.softmax(student_predictions, dim=1), true_labels)
    KL_loss = temperature ** 2 * LKL(torch.nn.functional.log_softmax(student_predictions, dim=1),
                                     torch.nn.functional.log_softmax(teacher_predictions, dim=1))
    CE_loss_T = LCE(torch.nn.functional.softmax(teacher_predictions, dim=1), true_labels)

    # compute gradients with respect to inputs,
    # use create graph option to make sure second order gradients can be computed with .backward()

    teacher_grad = torch.autograd.grad(CE_loss_T, inputs, retain_graph=True, create_graph=True)[0]
    student_grad = torch.autograd.grad(CE_loss, inputs, retain_graph=True, create_graph=True)[0]

    grad_discrepancy = torch.norm(teacher_grad - student_grad)
    relative_grad_discrepancy = grad_discrepancy / torch.norm(student_grad)
    return (lambda_CE * CE_loss + lambda_KL * KL_loss + lambda_GAD * grad_discrepancy,
            CE_loss, KL_loss, grad_discrepancy, relative_grad_discrepancy)


class Dummy_writer:
    def __init__(self):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def knowledge_distillation_training(distillation_dataset: Dataset, num_classes: int,
                                    teacher_model: nn.Module,
                                    student_model: nn.Module,
                                    batch_size: int = 128, epochs: int = 100, learn_rate=0.0005,
                                    Optimizer=optim.Adam, device="cpu",
                                    l_CE: float = 1., l_KD: float = 1., l_GAD: float = 1.,
                                    temperature=lambda x: np.exp(-x / 100),
                                    log_writer: SummaryWriter = Dummy_writer()):
    """
    performs the knowledge distillation training procedure.
    :param distillation_dataset:
    :param num_classes:
    :param teacher_model:
    :param student_model:
    :param batch_size:
    :param epochs:
    :param learn_rate:
    :param Optimizer:
    :param device:
    :param l_CE:
    :param l_KD:
    :param l_GAD:
    :param temperature:
    :param log_writer:
    :return: none, student model is trained in-place.
    """
    teacher_model.to(device)
    teacher_model.eval()
    student_model.to(device)
    optimizer = Optimizer(student_model.parameters(), lr=learn_rate)
    # Create DataLoader
    for epoch in tqdm(range(epochs), ncols=50 + epochs):
        train_loader = DataLoader(distillation_dataset, batch_size=batch_size, shuffle=True)

        for batch_idx, (inputs, _) in enumerate(train_loader):  # we deliberately ignore real labels!
            optimizer.zero_grad()
            inputs = inputs.to(device)
            inputs.requires_grad = True
            # Forward pass
            outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs)
            teacher_predictions = teacher_model(inputs)
            synthetic_labels = torch.eye(num_classes).to(device)[torch.argmax(teacher_predictions, dim=1)]
            loss, ce, kl, gad, grad_ratio = LGAD(inputs, synthetic_labels, outputs, teacher_outputs,
                                                 temperature=temperature(epoch), lambda_GAD=l_GAD,
                                                 lambda_CE=l_CE, lambda_KL=l_KD)

            log_writer.add_scalar("Loss/total_LGAD", loss.item(), epoch * len(train_loader) + batch_idx)
            log_writer.add_scalar("Loss/CE", ce.item(), epoch * len(train_loader) + batch_idx)
            log_writer.add_scalar("Loss/KL", kl.item(), epoch * len(train_loader) + batch_idx)
            log_writer.add_scalar("Loss/GAD", gad.item(), epoch * len(train_loader) + batch_idx)
            log_writer.add_scalar("Loss/grad_ratio", grad_ratio.item(), epoch * len(train_loader) + batch_idx)

            loss.backward()
            optimizer.step()
    log_writer.flush()
