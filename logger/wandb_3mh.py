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
        "scheduler": config.scheduler,
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
    wandb.define_metric("Train gender loss", summary="min")
    wandb.define_metric("Train age loss", summary="min")
    wandb.define_metric("Train mask f1-score", summary="max")
    wandb.define_metric("Train gender f1-score", summary="max")
    wandb.define_metric("Train age f1-score", summary="max")

    wandb.define_metric("Val avg loss", summary="min")
    wandb.define_metric("Val mask loss", summary="min")
    wandb.define_metric("Val gender loss", summary="min")
    wandb.define_metric("Val age loss", summary="min")
    wandb.define_metric("Val mask f1-score", summary="max")
    wandb.define_metric("Val gender f1-score", summary="max")
    wandb.define_metric("Val age f1-score", summary="max")
    print("wanbd init done.")

def logging(train_avg_loss, train_mask_loss, train_gender_loss, train_age_loss, train_mask_score, train_gender_score, train_age_score,
                val_avg_loss, val_mask_loss, val_gender_loss, val_age_loss, val_mask_score, val_gender_score, val_age_score):
    wandb.log({
    "Train avg loss": train_avg_loss,
    "Train mask loss": train_mask_loss,
    "Train gender loss": train_gender_loss,
    "Train age loss": train_age_loss,
    "Train mask f1-score": train_mask_score,
    "Train gender f1-score": train_gender_score,
    "Train age f1-score": train_age_score,
    "Val avg loss": val_avg_loss,
    "Val mask loss": val_mask_loss,
    "Val gender loss": val_gender_loss,
    "Val age loss": val_age_loss,
    "Val mask f1-score": val_mask_score,
    "Val gender f1-score": val_gender_score,
    "Val age f1-score": val_age_score
    })

def finish():
    wandb.finish()