import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from RobustMockTeacher import MockNeuralNetwork
from src.utils.Relu_network import FFNetwork
from src.utils.data.datasets import SyntheticDataset
from src.utils.distillation.knowledge_distillation import knowledge_distillation_training

from src.utils.metrics import robustness_disparity

writer = SummaryWriter()


def high_confidence_data(synthetic_data, model, confidence):
    mask = (torch.softmax(model(synthetic_data), dim=1).max(dim=1).values > confidence)
    return synthetic_data[mask, :], mask


def high_confidence_indices(data, model, confidence):
    mask = (torch.softmax(model(data), dim=1).max(dim=1).values > confidence)
    return mask


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    n = 50000
    dim = 4
    # Load and preprocess dataset
    print(np.random.uniform(size=(n, dim), low=-1, high=1))
    data = torch.Tensor(np.random.uniform(size=(n, dim), low=-1, high=1))
    print(data[:, 0])
    teacher = MockNeuralNetwork(num_dim=dim, frequency=500)
    # Distill student.

    student = FFNetwork(input_dim=dim, output_dim=2, layer_sizes=[5, 5])

    # Knowledge distillation.
    print("Performing Knowledge Distillation")
    noise_radius = 1
    distillation_data = SyntheticDataset(data=data, labels=(teacher(data)[:, 0] > 0.5).type(torch.long))

    l_GAD = 50
    l_CE = 2
    l_KD = 5
    epochs = 50

    knowledge_distillation_training(distillation_data, 2, teacher, student,
                                    l_GAD=l_GAD, l_CE=l_CE, l_KD=l_KD, learn_rate=0.0001, epochs=epochs,
                                    data_loader_workers=0)
    writer.close()
    mask = high_confidence_indices(distillation_data.features, student, confidence=0.8)
    rob_disparity, student_grad = robustness_disparity(distillation_data, teacher, student)
    # rob_disparity = rob_disparity[mask]
    # student_grad = student_grad[mask]
    print(mask.type(torch.float).mean())
    _, ind = rob_disparity.min(dim=0)
    # Checking for disagreement.
    print(f"min relative robustness disparity: {rob_disparity.min()}\n"
          f"teacher_grad at that point: {student_grad[ind]}\n"
          f"mean relative robustness disparity: {rob_disparity.mean()}\n"
          f"median relative robustness disparity: {rob_disparity.median()}\n"
          f"quantile1 relative robustness disparity: {rob_disparity.quantile(q=.01)}\n")
