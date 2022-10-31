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
        "scheduler": config.scheduler,
        "losses": config.losses,
        "loss_rate": config.loss_rate,
        "split_rate": config.split_rate,
        "scheduler": "ReduceLROnPlateau",
        "image_size": config.image_size,
        "crop_size": config.crop_size,
        "mean": config.mean,
        "std": config.std
        }

    wandb.init(project="MaskStatusClassification", entity="wowns1484", config=wandb_config, reinit=True)
    wandb.run.name = config.desc
    wandb.run.save()

    wandb.define_metric("Train avg loss", summary="min")
    wandb.define_metric("Train mask loss", summary="min")
    wandb.define_metric("Train age gender loss", summary="min")
    wandb.define_metric("Train mask F1-Score", summary="max")
    wandb.define_metric("Train age gender F1-Score", summary="max")
    wandb.define_metric("Val avg loss", summary="min")
    wandb.define_metric("Val mask loss", summary="min")
    wandb.define_metric("Val age gender loss", summary="min")
    wandb.define_metric("Val mask F1-Score", summary="max")
    wandb.define_metric("Val age gender F1-Score", summary="max")
    print("wanbd init done.")

def logging(train_avg_loss, train_mask_loss, train_age_gender_loss, train_mask_score, train_age_gneder_score, 
            val_avg_loss, val_mask_loss, val_age_gender_loss, val_mask_score, val_age_gender_score):
    wandb.log({
    "Train avg Loss": train_avg_loss,
    "Train mask Loss": train_mask_loss,
    "Train age gender Loss": train_age_gender_loss,
    "Train mask F1-Score": train_mask_score,
    "Train age gender F1-Score": train_age_gneder_score,
    "Val avg Loss": val_avg_loss,
    "Val mask Loss": val_mask_loss,
    "Val age gender Loss": val_age_gender_loss,
    "Val mask F1-Score": val_mask_score,
    "Val age gender F1-Score": val_age_gender_score,
    })

def finish():
    wandb.finish()