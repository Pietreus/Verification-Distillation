import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utils.Relu_network import FFNetwork


class SimpleNN(nn.Module):
    def __init__(self, layers):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


def read_nnet_file(file_path):
    with open(file_path, 'r') as f:
        # Skip header lines starting with "//"
        line = f.readline()
        while line.startswith("//"):
            line = f.readline()
        num_layers, num_inputs, num_outputs, max_layer_size = map(int, line.strip().split(','))

        line = f.readline()
        # Read layer sizes
        layer_sizes = [int(size) for size in line.strip().split(',')]

        # Skip unused flag
        f.readline()

        # Read normalization values (can be ignored for building the model, but necessary for context)
        input_mins = np.array([float(x) for x in f.readline().strip().split(",")])
        input_maxs = np.array([float(x) for x in f.readline().strip().split(",")])
        input_means = np.array([float(x) for x in f.readline().strip().split(",")])
        input_ranges = np.array([float(x) for x in f.readline().strip().split(",")])

        # Read weights and biases
        weights = []
        biases = []
        for i in range(num_layers):
            # Read weights
            weight_matrix = []
            for _ in range(layer_sizes[i + 1]):
                weight_matrix.append([float(x) for x in f.readline().strip().split(",")])
            print(weight_matrix)
            weights.append(np.array(weight_matrix))

            # Read biases
            bias_vector = []
            for _ in range(layer_sizes[i + 1]):
                bias_vector.append([float(x) for x in f.readline().strip().split(",")])
            print(bias_vector)
            biases.append(np.array([elem for sublist in bias_vector for elem in sublist]))

    return num_inputs, num_outputs, layer_sizes, weights, biases


def build_pytorch_model(input_size, output_size, layer_sizes, weights, biases):
    model = SimpleNN(layer_sizes)
    for i, layer in enumerate(model.layers):
        layer.weight = nn.Parameter(torch.tensor(weights[i], dtype=torch.float32))
        layer.bias = nn.Parameter(torch.tensor(biases[i], dtype=torch.float32))
    return model


def load_pytorch_model_from_file(file_path,model):
    """
    Loads a PyTorch model from a .pt file.

    Args:
    - file_path (str): Path to the .pt file containing the model.

    Returns:
    - model (torch.nn.Module): The loaded PyTorch model.
    """
    state_dict = torch.load(file_path)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    return model
def main():
    # File path to the .nnet file
    file_path = "../../../Global_2Safety_with_Confidence/104k_teacher_noise_1_CE_2_KL_5_GAD_50.nnet"

    # Read the .nnet file
    input_size, output_size, layer_sizes, weights, biases = read_nnet_file(file_path)
    # Build the PyTorch model
    # model = FFNetwork(input_size, output_size, layer_sizes[1:len(layer_sizes)-1])
    # print(model)
    print(f"{input_size}, {layer_sizes}, {output_size}")
    # model = load_pytorch_model_from_file("../../../models/student_noise_1_CE_2_KL_5_GAD_50.pt", model)
    model = build_pytorch_model(input_size, output_size, layer_sizes, weights, biases)
    # Example input vector (must match the input size)
    input_vector = torch.tensor([1.586678, 1.0, 1.271822, 1.00001, 2.212861, 1.696399, 1.0, 1.0], dtype=torch.float32)

    # Run the model to get the raw prediction
    model.eval()
    with torch.no_grad():
        raw_prediction = model(input_vector)

    # Apply softmax to get probabilities
    softmax_prediction = F.softmax(raw_prediction, dim=0)

    # Print the results
    print("Raw Prediction:", raw_prediction)
    print("Softmax Prediction:", softmax_prediction)


if __name__ == "__main__":
    main()
