from torch import nn

from src.utils.data.datasets import get_loaders, NoisyDataset
from src.utils.knowledge_distillation import knowledge_distillation_training

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student_model.parameters(), lr=0.0001)

# Parameters.
l_GAD = 5
l_CE = 2
l_KD = 5
epochs = 100

input_dim = 4
output_dim = 3

train_loader, val_loader, test_loader = get_loaders('iris', val_split=0.0, batch_size=16)
noisy_dataset = NoisyDataset(train_loader.dataset, copies_per_sample=2000, noise_std=0.1)
knowledge_distillation_training(noisy_dataset, num_classes=output_dim, teacher_model=teacher_model,
                                student_model=student_model, device="cpu")

student_rmodel = mair.RobModel(student_model, n_classes=output_dim)
evaluate_and_print(student_rmodel, "student", eps=EPS, std=STD)
