import torch
import numpy as np
from torch import nn

from KD import knowledge_distillation
from RobustMockTeacher import MockNeuralNetwork
from src.Utils import nnet_exporter


class StudentModel(nn.Module):
    def __init__(self, input_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 3)
        self.fc6 = nn.Linear(3, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc6(x)
        return x

def high_confidence_data(synthetic_data, model, confidence):
    mask = torch.flatten(torch.abs(model.raw_output(synthetic_data)) > confidence, 0)
    return synthetic_data[mask]

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    dim = 2
    delta_radius = 0.05
    device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"

    for frequency in np.linspace(0.10, 0.56, 1):

        losses = []

        for l_GAD in np.linspace(0.5, 4, 5):
            for l_CE in np.linspace(1.5, 4, 5):
                for l_KD in np.linspace(1, 1, 5):
                    student = StudentModel(dim)
                    mock_teacher = MockNeuralNetwork(dim, frequency, device=device)
                    student.to(device)
                    mock_teacher.to(device)
                    # Generate synthetic data using normal distribution
                    synthetic_data = torch.tensor(np.random.uniform(-2, 2, size=(10*(10 ** 4), *(dim,))), dtype=torch.float32).to(device)
                    # synthetic_data = high_confidence_data(synthetic_data, mock_teacher, confidence=0.6)
                    print(f"teacher frequency: {frequency}  "
                          f"teacher {delta_radius}-robustness: {mock_teacher.data_robustness(synthetic_data, delta_radius):0.3f}")

                    loss = knowledge_distillation(synthetic_data, mock_teacher, student, batch_size=1000, epochs=50,
                                                           print_functions=True, device=device,
                                                           l_GAD=l_GAD, l_CE=l_CE, l_KD=l_KD,
                                                  confidence=0.7)
                    torch.onnx.export(student, args=(synthetic_data[1]),
                                      f=f"models/dim_{dim}_delta_{delta_radius}_frequency_{frequency}.onnx")
                    print(loss, l_GAD, l_CE, l_KD)
                    torch.save(student.state_dict(), f"models/dim_{dim}_delta_{delta_radius}_frequency_{frequency}.pt")
                    nnet_exporter(student,f"models/dim_{dim}_delta_{delta_radius}_frequency_{frequency}.nnet",)
                    losses.append(loss)


        # confidence is just a threshold use softmax of outputs
        # for distillation reject samples with low confidence
        # german credit dataset