import mair
import numpy as np
import torch
import yaml
from mair import AT
import wandb

from src.utils.Relu_network import FFNetwork
from src.utils.data.datasets import get_loaders, NoisyDataset
from src.utils.knowledge_distillation import knowledge_distillation_training_wandb

input_dim = 768
output_dim = 10


def distilling_search(trainer, train_loader, val_loader, test_loader):
    wandb.init(entity="peter-blohm-tu-wien", project="garbage/mnist")
    np.random.seed(wandb.config.seed)
    torch.random.manual_seed(wandb.config.seed)
    student_model = FFNetwork(input_dim, output_dim, layer_sizes=wandb.config.layer_sizes)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    noisy_dataset = NoisyDataset(train_loader.dataset, copies_per_sample=wandb.config.copies, noise_std=0.1)
    knowledge_distillation_training_wandb(noisy_dataset, num_classes=output_dim, teacher_model=trainer,
                                          student_model=student_model, device=device, val_loader=val_loader,
                                          l_CE=wandb.config.l_CE, l_KD=wandb.config.l_KD,
                                          l_GAD=wandb.config.l_GAD, learn_rate=wandb.config.learn_rate,
                                          epochs=wandb.config.epochs,
                                          batch_size=wandb.config.batch_size,
                                          test_loader=test_loader)


if __name__ == "__main__":

    with open("mair_experiment/mnist/distillation_search.yaml", 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)
    #TODO parameters for teacher + grid
    train_loader, val_loader, test_loader = get_loaders('mnist', val_split=0.2, batch_size=8)
    teacher_model = FFNetwork(input_dim, output_dim, layer_sizes=[10, 10])
    rmodel = mair.RobModel(teacher_model, n_classes=output_dim)  # .cuda
    trainer = AT(rmodel, eps=.1, alpha=.1, steps=10)
    trainer.record_rob(train_loader, val_loader, eps=.1, alpha=.1, steps=10, std=0.1)
    trainer.setup(optimizer="SGD(lr=0.1, momentum=0.9)",
                  scheduler="Step(milestones=[100, 150], gamma=0.1)",
                  scheduler_type="Epoch",
                  minimizer=None,  # or "AWP(rho=5e-3)",
                  n_epochs=100
                  )
    trainer.fit(train_loader=train_loader,
                n_epochs=100,
                save_path='../rob/',
                save_best={"Clean(Val)": "HBO", "PGD(Val)": "HB"},
                save_type="Epoch",
                save_overwrite=True,
                record_type="Epoch"
                )

    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="garbage/mnist", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: distilling_search(teacher_model, train_loader, val_loader, test_loader))
