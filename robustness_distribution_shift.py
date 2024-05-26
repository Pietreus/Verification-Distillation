import csv
import os
import socket
from datetime import datetime
import concurrent.futures


import torch
import torch.multiprocessing as mp
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

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def high_confidence_data(synthetic_data, model, confidence):
    mask = (torch.abs(torch.softmax(model(synthetic_data), dim=1) - 0.5) * 2 > confidence)[:, 0]
    return synthetic_data[mask, :]


def all_high_confidence_data_delta_robust(synthetic_data, model, confidence, delta):
    hi_conf_data = high_confidence_data(synthetic_data, model, confidence)
    if len(hi_conf_data) == 0:
        return np.nan

    return np.all(np.abs(np.sum(hi_conf_data.numpy(), axis=1)) > delta)


def all_high_confidence_data_robustness_radius(synthetic_data, model, confidence):
    hi_conf_data = high_confidence_data(synthetic_data, model, confidence)
    if len(hi_conf_data) == 0:
        return np.nan
    return np.min(np.abs(np.sum(hi_conf_data.numpy(), axis=1)))


def generate_normal_matrix(center, variance, shape, support_min, support_max, distance_cutoff=2):
    """
    Generates a normally distributed numpy matrix with specified center, variance,
    and resamples values to ensure they fall within the specified support range.

    Parameters:
    center (float): The mean value for the normal distribution.
    variance (float): The variance for the normal distribution.
    shape (tuple): The shape of the output matrix.
    support_min (float): The minimum value in the support range.
    support_max (float): The maximum value in the support range.

    Returns:
    np.ndarray: A numpy matrix with specified properties.
    """
    # Calculate standard deviation from variance
    std_dev = np.sqrt(variance)

    # Create an empty matrix to fill
    matrix = np.empty(shape)

    # Generate elements until all are within the support range
    num_elements = shape[0]
    count = 0

    while count < num_elements:
        # Generate candidate points
        samples = np.random.normal(loc=center, scale=std_dev,
                                   size=((1 + int(np.sqrt(variance))) * num_elements - count + 1, shape[1]))

        # Filter points within the support range
        support_valid_samples = samples[(np.sum(samples, axis=1) >= 0) &
                                        (np.all(samples >= support_min, axis=1)) &
                                        (np.all(samples <= support_max, axis=1)) &
                                        (np.linalg.norm(samples - center, axis=1) <= distance_cutoff), :]
        # diag_valid_samples = samples[(np.sum(samples,axis=1) >= 0), :]
        # Determine how many valid samples we can use
        num_valid = min(len(support_valid_samples), shape[0] - count)

        # Place valid samples in the matrix
        matrix[count:count + num_valid, :] = support_valid_samples[:num_valid, :]

        # Filter points to be above the diagonal
        # matrix = matrix[(np.sum(matrix, axis=1) >= 0),:]
        count += num_valid

    return matrix









if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    variance = 0.5 ** 2
    n = 5 * 10 ** 4
    dim = 2
    l_GAD = 500
    l_CE = 5
    l_KD = 1
    confidence = 0.5
    frequency = 100
    delta_radius = 0.05
    device = "cpu"

    synthetic_data = np.vstack((
        generate_normal_matrix(0.707, variance, (n, dim), -8, 8, distance_cutoff=2.5 * np.sqrt(variance)),
        -generate_normal_matrix(0.707, variance, (n, dim), -8, 8, distance_cutoff=2.5 * np.sqrt(variance))))
    synthetic_data = torch.tensor(synthetic_data, dtype=torch.float32).to(device)
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    # Get the hostname: e.g., thinkpad
    hostname = socket.gethostname()
    # Combine to form the full log directory path: e.g., my_subdir/May26_13-07-26_thinkpad
    csv_file = os.path.join("runs", f"{current_time}_{hostname}", "summary.csv")
    data = []
    processes = []
    confidences = [0.5, 0.6, 0.7]
    frequencies = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 75, 80, 85, 90, 95, 100, 150, 200]
    for frequency in frequencies:
        for confidence in confidences:
            student = StudentModel(dim)
            mock_teacher = MockNeuralNetwork(dim, frequency, device=device)
            student.to(device)
            mock_teacher.to(device)
            # Generate synthetic data using normal distribution
            # synthetic_data = high_confidence_data(synthetic_data, mock_teacher, confidence=0.6)
            # print(f"teacher frequency: {frequency}  "
            #       f"teacher {delta_radius}-robustness: {mock_teacher.data_robustness(synthetic_data, delta_radius):0.3f}")

            loss, grad_ratio = knowledge_distillation(synthetic_data, mock_teacher, student, batch_size=1000, epochs=20,
                                                      print_functions=True, device=device,
                                                      l_GAD=l_GAD, l_CE=l_CE, l_KD=l_KD,
                                                      confidence=confidence, teacher_freq=frequency)
            teacher_robust = all_high_confidence_data_delta_robust(synthetic_data, mock_teacher, confidence,
                                                                   delta_radius)
            teacher_robustness_radius = all_high_confidence_data_robustness_radius(synthetic_data, mock_teacher,
                                                                                   confidence)

            student_robust = all_high_confidence_data_delta_robust(synthetic_data, student, confidence, delta_radius)
            student_robustness_radius = all_high_confidence_data_robustness_radius(synthetic_data, student, confidence)
            print(f"frequency: {frequency:.6f} "
                  f"variance: {variance:.6f} confidence: {confidence}, delta: {delta_radius}"
                  f" teacher robust: {teacher_robust}"
                  f"({teacher_robustness_radius:.6f})"
                  f" student robust: {student_robust}"
                  f"({student_robustness_radius:.6f})"
                  f" gradient ratio: {grad_ratio}")
            data.append((confidence, frequency, delta_radius, variance, l_GAD, l_CE, l_KD, dim, n,
                       teacher_robust, teacher_robustness_radius, student_robust, student_robustness_radius,
                       grad_ratio.detach().numpy()))
            # writer = SummaryWriter()

    header = ["confidence", "frequency", "delta_radius", "variance", "l_GAD", "l_CE", "l_KD", "dim", "n",
              "teacher_robust", "teacher_robustness_radius", "student_robust", "student_robustness_radius", "grad_ratio"]
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

    print(f"Data successfully written to {csv_file}")
