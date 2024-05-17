import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

import KD
from RobustMockTeacher import MockNeuralNetwork


# Define the teacher model
# Define TeacherModel
class TeacherModel(nn.Module):
    def __init__(self, input_size):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the student model
class StudentModel(nn.Module):
    def __init__(self, input_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)
        self.fc2 = nn.Linear(1, 1)
        self.fc3 = nn.Linear(1, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x




def train_and_plot_progress(history):

    # Plot training & validation accuracy values
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    # Generate binary classification data
    X, y = make_classification(n_samples=10 ** 4, n_features=5, n_classes=2, random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train_2 = torch.eye(2)[y_train]
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # Create the model
    teacher = TeacherModel(X_train.shape[1])

    # Print the model to ensure it is defined correctly
    print(list(teacher.parameters()))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(teacher.parameters(), lr=0.1, momentum=0.9)
    print("Optimizer created successfully")
    # Train the model and plot the progress
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    print("Teacher training")
    for epoch in tqdm(range(10)):
        teacher.train()
        optimizer.zero_grad()
        outputs = teacher(X_train)
        # print(outputs)
        loss = criterion(outputs, y_train_2)
        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['accuracy'].append((outputs.argmax(dim=1) == y_train).float().mean().item())

        # Validation
        with torch.no_grad():
            teacher.eval()
            outputs_val = teacher(X_train)
            loss_val = criterion(outputs_val, torch.tensor(y_train_2, dtype=torch.float32))
            history['val_loss'].append(loss_val.item())
            history['val_accuracy'].append((outputs_val.argmax(dim=1) == y_train).float().mean().item())

    train_and_plot_progress(history)

    student = StudentModel(X_train.shape[1])
    mockTeacher = MockNeuralNetwork(5, 0.02)
    history = KD.knowledge_distillation(mockTeacher, student, 10 ** 4, (5,), 100, 100)
    train_and_plot_progress(history)
