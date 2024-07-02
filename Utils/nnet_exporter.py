import csv

import torch
from torch import nn


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


# https://github.com/sisl/NNet
def nnet_exporter(model: torch.nn.Module, file, inputs: torch.tensor,  out_mean: int, out_range: int, comment: str = ""):

    model_dict = model.state_dict()

    # assume/hope this works :)
    weights = [value for name, value in model_dict.items() if name.startswith("fc") and name.endswith("weight")]
    biases = [value for name, value in model_dict.items() if name.startswith("fc") and name.endswith("bias")]

    with open(file, 'w', newline='\n') as f:
        # 1: Header text. This can be any number of lines so long as they begin with "//"
        f.write('//' + comment + "\n")
        # 2: Four values: Number of layers, number of inputs, number of outputs, and maximum layer size
        f.write(f"{len(weights)}, "
                        f"{weights[0].shape[1]}, "
                        f"{weights[-1].shape[0]}, "
                        f"{max([layer.shape[1] for layer in weights])}\n")
        # 3: A sequence of values describing the network layer sizes. Begin with the input size, then the size of the first layer, second layer, and so on until the output layer size
        f.write(str(weights[0].shape[1]) + ", " + ", ".join([str(layer.shape[0]) for layer in weights]) +"\n")
        # 4: A flag that is no longer used, can be ignored
        f.write("0\n")
        # 5: Minimum values of inputs (used to keep inputs within expected range)
        f.write(f"{", ".join([str(inp.item()) for inp in inputs.min(dim=0).values])}\n")
        # 6: Maximum values of inputs (used to keep inputs within expected range)
        f.write(f"{", ".join([str(inp.item()) for inp in inputs.max(dim=0).values])}\n")
        # 7: Mean values of inputs and one value for all outputs (used for normalization)
        f.write(f"{", ".join([str(inp.item()) for inp in inputs.mean(dim=0)])}" + f", {str(out_mean.item())}\n")
        # 8: Range values of inputs and one value for all outputs (used for normalization)
        f.write(f"{", ".join([str(inp.item()) for inp in (inputs.max(dim=0).values - inputs.min(dim=0).values)])}" + f", {str(out_range.item())}\n")
        # 9+: Begin defining the weight matrix for the first layer, followed by the bias vector.
        # The weights and biases for the second layer follow after, until the weights and biases for the output layer are defined.
        for weight, bias in zip(weights, biases):
            for row in weight:
                f.write(" ".join([str(r.item()) for r in row]) + "\n")
            for value in bias:
                f.write(f"{value}\n")


if __name__ == "__main__":
    teacher = TeacherModel(3, 2)
    input = torch.rand(3, 2000)
    nnet_exporter(teacher, "teacher_model.nnet", input, 2, 2)