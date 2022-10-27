import wandb

## key: a9c5b57ab001c02847570b604a59f807ebc65a47

def init_wandb(config : dict):
    wandb_config = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "model_name": config.model_name,
        "optimizer": config.optimizer,
        "loss": config.loss,
        "split_rate": config.split_rate,
        "scheduler": "ReduceLROnPlateau",
        "image_size": config.image_size,
        "mean": config.mean,
        "std": config.std
        }

    wandb.init(project="MaskStatusClassification", entity="wowns1484", config=wandb_config, reinit=True)
    wandb.run.name = config.desc
    wandb.run.save()

    wandb.define_metric("Train avg loss", summary="min")
    wandb.define_metric("Train F1-Score", summary="max")
    wandb.define_metric("Validation avg loss", summary="min")
    wandb.define_metric("Validation F1-Score", summary="max")
    print("wanbd init done.")

def logging(train_score, train_avg_loss, val_score, validation_avg_loss):
    wandb.log({
    "Train F1-Score": train_score,
    "Train Avg Loss": train_avg_loss,
    "Validation F1-Score": val_score,
    "Validation Avg Loss": validation_avg_loss
    })

def finish():
    wandb.finish()