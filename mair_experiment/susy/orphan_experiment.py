import warnings

import mair
import numpy as np
import torch
import yaml
import wandb
from mair import AT, Standard

from src.utils.data.datasets import get_loaders
from src.utils.Relu_network import FFNetwork

warnings.simplefilter("ignore")


def orphan_training():
    # Set seeds.
    wandb.init(entity="peter-blohm-tu-wien", project="distillation_orphans")
    np.random.seed(wandb.config.seed)
    torch.random.manual_seed(wandb.config.seed)
    print(wandb.config)
    train_loader, val_loader, test_loader = get_loaders('susy', val_split=0.0, batch_size=wandb.config.batch_size)

    input_dim = 4
    output_dim = 3

    orphan_model = FFNetwork(input_dim, output_dim, layer_sizes=wandb.config.layer_sizes)

    # Robust orphan training.
    robust_orphan_rmodel = mair.RobModel(orphan_model, n_classes=output_dim)  #.cuda()
    if torch.cuda.is_available():
        robust_orphan_rmodel = robust_orphan_rmodel.cuda()
    wandb.watch(robust_orphan_rmodel)
    if wandb.config.optimization_function == "AT":
        robust_orphan_trainer = AT(robust_orphan_rmodel,
                                   eps=wandb.config.EPS,
                                   alpha=wandb.config.ALPHA,
                                   steps=wandb.config.STEPS)
    else:
        robust_orphan_trainer = Standard(robust_orphan_rmodel)
    robust_orphan_trainer.record_rob(train_loader, val_loader, eps=wandb.config.EPS, alpha=wandb.config.ALPHA,
                                     steps=wandb.config.STEPS, std=wandb.config.STD)
    robust_orphan_trainer.setup(optimizer=f"SGD(lr={wandb.config.lr}, momentum={wandb.config.momentum})",
                                scheduler="Step(milestones=[100, 150], gamma=0.1)",
                                scheduler_type="Epoch",
                                minimizer=None,  # or "AWP(rho=5e-3)",
                                n_epochs=wandb.config.n_epochs,
                                )
    robust_orphan_trainer.fit(train_loader=train_loader,
                              n_epochs=wandb.config.n_epochs,
                              save_path='../rob/',
                              save_best={"Clean(Val)": "HBO", "PGD(Val)": "HB"},
                              save_type="Epoch",
                              save_overwrite=True,
                              record_type="Epoch"
                              )
    wandb.log(
        {"train-acc": robust_orphan_rmodel.eval_accuracy(train_loader),
         "train-GN-rob": robust_orphan_rmodel.eval_rob_accuracy_gn(train_loader, std=wandb.config.STD),
         "train-FGSM robustness": robust_orphan_rmodel.eval_rob_accuracy_fgsm(train_loader, eps=wandb.config.EPS),
         "val-acc": robust_orphan_rmodel.eval_accuracy(val_loader),
         "val-GN-rob": robust_orphan_rmodel.eval_rob_accuracy_gn(val_loader, std=wandb.config.STD),
         "val-FGSM robustness": robust_orphan_rmodel.eval_rob_accuracy_fgsm(val_loader, eps=wandb.config.EPS),
         "test-acc": robust_orphan_rmodel.eval_accuracy(test_loader),
         "test-GN-rob": robust_orphan_rmodel.eval_rob_accuracy_gn(test_loader, std=wandb.config.STD),
         "test-FGSM robustness": robust_orphan_rmodel.eval_rob_accuracy_fgsm(test_loader, eps=wandb.config.EPS)})

    # Saving models locally and as an artifact.
    torch.save(robust_orphan_rmodel.state_dict(), f"{wandb.run.dir}.h5")

    artifact = wandb.Artifact(sweep_id, type='model')  # wandb.run.id
    artifact.add_file(f"{wandb.run.dir}.h5")
    wandb.run.log_artifact(artifact, aliases=[wandb.run.id])


if __name__ == "__main__":
    with open("mair_experiment/sweep.yaml", 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)

    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="distillation-orphans", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=orphan_training)
