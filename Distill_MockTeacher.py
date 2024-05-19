from torch import nn

from KD import knowledge_distillation
from RobustMockTeacher import MockNeuralNetwork


class StudentModel(nn.Module):
    def __init__(self, input_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    dim = 2
    student = StudentModel(dim)
    mock_teacher = MockNeuralNetwork(dim, 0.5)
    knowledge_distillation(mock_teacher, student, 10 ** 5, (dim,), 100, 100, True)
