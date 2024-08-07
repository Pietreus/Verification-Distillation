import mair
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mair import Standard
from mair.defenses import AT
from tqdm import tqdm

from KD import LGAD
from src.utils.knowledge_distillation import knowledge_distillation_training
from teacher_student_distillation import writer
from src.utils.data.datasets import get_loaders, get_noisy_loader, get_input_output_dimensions, NoisyDataset
from src.utils.Relu_network import FFNetwork

train_loader, val_loader, test_loader = get_loaders('iris', val_split=0.0, batch_size=16)
noisy_dataset = NoisyDataset(train_loader.dataset, copies_per_sample=2000, noise_std=0.1)
# noisy_train_loader = get_noisy_loader(train_loader, copies_per_sample=20, noise_std=0.1)


# input_dim, output_dim = get_input_output_dimensions(train_loader)
input_dim = 4
output_dim = 3

teacher_model = FFNetwork(input_dim, output_dim, layer_sizes=[10, 10])
orphan_model = FFNetwork(input_dim, output_dim, layer_sizes=[10])
robust_orphan_model = FFNetwork(input_dim, output_dim, layer_sizes=[10])
student_model = FFNetwork(input_dim, output_dim, layer_sizes=[10])


# Variables.
EPS = 0.1
ALPHA = 0.1
STEPS = 10
STD = 0.1
n_epochs = 10


# Teacher training.
rmodel = mair.RobModel(teacher_model, n_classes=output_dim)#.cuda
trainer = AT(rmodel, eps=EPS, alpha=ALPHA, steps=STEPS)
trainer.record_rob(train_loader, val_loader, eps=EPS, alpha=ALPHA, steps=STEPS, std=STD)
trainer.setup(optimizer="SGD(lr=0.1, momentum=0.9)",
              scheduler="Step(milestones=[100, 150], gamma=0.1)",
              scheduler_type="Epoch",
              minimizer=None, # or "AWP(rho=5e-3)",
              n_epochs=n_epochs
              )
trainer.fit(train_loader=train_loader,
            n_epochs=n_epochs,
            save_path='../rob/',
            save_best={"Clean(Val)":"HBO", "PGD(Val)":"HB"},
            save_type="Epoch",
            save_overwrite=True,
            record_type="Epoch"
            )

# Robust orphan training.
robust_orphan_rmodel = mair.RobModel(robust_orphan_model, n_classes=output_dim)#.cuda()
robust_orphan_trainer = AT(robust_orphan_rmodel, eps=EPS, alpha=ALPHA, steps=STEPS)
robust_orphan_trainer.record_rob(train_loader, val_loader, eps=EPS, alpha=ALPHA, steps=STEPS, std=STD)
robust_orphan_trainer.setup(optimizer="SGD(lr=0.1, momentum=0.9)",
                     scheduler="Step(milestones=[100, 150], gamma=0.1)",
                     scheduler_type="Epoch",
                     minimizer=None, # or "AWP(rho=5e-3)",
                     n_epochs=n_epochs
                     )
robust_orphan_trainer.fit(train_loader=train_loader,
                          n_epochs=n_epochs,
                          save_path='../rob/',
                          save_best={"Clean(Val)":"HBO", "PGD(Val)":"HB"},
                          save_type="Epoch",
                          save_overwrite=True,
                          record_type="Epoch"
                          )

# Orphan training.
orphan_rmodel = mair.RobModel(orphan_model, n_classes=output_dim)#.cuda()
orphan_trainer = Standard(orphan_rmodel)
orphan_trainer.record_rob(train_loader, val_loader, eps=EPS, alpha=ALPHA, steps=STEPS, std=STD)
orphan_trainer.setup(optimizer="SGD(lr=0.1, momentum=0.9)",
                     scheduler="Step(milestones=[100, 150], gamma=0.1)",
                     scheduler_type="Epoch",
                     minimizer=None, # or "AWP(rho=5e-3)",
                     n_epochs=n_epochs
                     )

orphan_trainer.fit(train_loader=train_loader,
                   n_epochs=n_epochs,
                   save_path='../rob/',
                   save_best={"Clean(Val)":"HBO", "PGD(Val)":"HB"},
                   # 'save_best': model with high PGD are chosen,
                   # while in similar cases, model with high Clean are selected.
                   save_type="Epoch",
                   save_overwrite=True,
                   record_type="Epoch"
                   )


def evaluate_and_print(model, label, std, eps):
    print("\n")
    print(f"Model: {label}")
    print(f"Clean accuracy: {model.eval_accuracy(val_loader):.2f}")  # clean accuracy
    print(f"GN robustness: {model.eval_rob_accuracy_gn(val_loader, std=std):.2f}")  # gaussian noise accuracy
    print(f"FGSM robustness: {model.eval_rob_accuracy_fgsm(val_loader, eps=eps):.2f}")  # FGSM accuracy
    print("\n")


evaluate_and_print(rmodel, "teacher", std=STD, eps=EPS)
evaluate_and_print(robust_orphan_rmodel, "robust orphan", std=STD, eps=EPS)
evaluate_and_print(orphan_rmodel, "orphan", std=STD, eps=EPS)

print("\n Distillation\n")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student_model.parameters(), lr=0.0001)

# Parameters.
l_GAD = 5
l_CE = 2
l_KD = 5
epochs = 100

knowledge_distillation_training(noisy_dataset, num_classes=output_dim, teacher_model=teacher_model,
                                student_model=student_model, device="cpu")

student_rmodel = mair.RobModel(student_model, n_classes=output_dim)
evaluate_and_print(student_rmodel, "student", eps=EPS, std=STD)