import wandb
import random  # for demo script
from modular.train import create_model
import torch


def load_and_draw(model_path, model_base, model_choice, schedule_lr):
    wandb.login()

    run = wandb.init(
        # Set the project where this run will be logged
        project="view_truck",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 0.0003,
            "architecture": 'CNN',
            "dataset": 'CIFAR-100',
            "epochs": 20,
        },
    )
    model, optimizer, loss_fn, scheduler = create_model(model_choice, model_base, schedule_lr)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss_test = checkpoint['loss_test']
    loss_train = checkpoint['loss_train']
    accuracy_test = checkpoint['accuracy_test']
    accuracy_train = checkpoint['accuracy_train']
    lrs = checkpoint['lrs']

    print(type(loss_train))
    print(type(lrs))


    model.eval()

    # simulating a training run
    for epoch in range(0, len(loss_train)):
        print(f"epoch={epoch+1}, "
              f"accuracy_test={accuracy_test[epoch]}, "
              f"accuracy_train={accuracy_train[epoch]}, "
              f"loss_test={loss_test[epoch]}, "
              f"loss_train={loss_train[epoch]}, "
              f"lr={lrs[epoch]}")

        wandb.log({"accuracy_test": accuracy_test[epoch],
                   "accuracy_train": accuracy_train[epoch],
                   "loss_test": loss_test[epoch],
                   "loss_train": loss_train[epoch],
                   "lr": lrs[epoch]})




# init_wandb(0.001, 'CNN', 'CIFAR-100', 20)
