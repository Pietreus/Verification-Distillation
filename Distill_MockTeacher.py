import torch
import numpy as np
from torch import nn

from KD import knowledge_distillation
from RobustMockTeacher import MockNeuralNetwork


class StudentModel(nn.Module):
    def __init__(self, input_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    dim = 2
    delta_radius = 0.05
    for frequency in np.linspace(0.06, 0.56, 26):
        student = StudentModel(dim)
        mock_teacher = MockNeuralNetwork(dim, frequency)
        # Generate synthetic data using normal distribution
        synthetic_data = torch.tensor(np.random.uniform(-2, 2, size=(10 ** 5, *(dim,))), dtype=torch.float32)
        print(f"teacher frequency: {frequency}  "
              f"teacher {delta_radius}-robustness: {mock_teacher.data_robustness(synthetic_data, delta_radius):0.3f}")

        knowledge_distillation(synthetic_data, mock_teacher, student, 1000, 1, True)
        torch.onnx.export(student, args=(synthetic_data[1]),
                          f=f"models/dim_{dim}_delta_{delta_radius}_frequency_{frequency}.onnx")
        torch.save(student.state_dict(), f"models/dim_{dim}_delta_{delta_radius}_frequency_{frequency}.pt")
