import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

import wandb
from sklearn.metrics import f1_score
import yaml
from easydict import EasyDict


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(
        figsize=(12, 18 + 2)
    )  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join(
            [
                f"{task} - gt: {gt_label}, pred: {pred_label}"
                for gt_label, pred_label, task in zip(gt_decoded_labels, pred_decoded_labels, tasks)
            ]
        )

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def weighted_loss(loss_list, weight_list):
    weighted_loss = 0
    for idx, loss in enumerate(loss_list):
        weighted_loss += loss * weight_list[idx]
    return weighted_loss


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    # save_dir = increment_path(os.path.join(model_dir, args.name))
    args.experiment_name = "_".join(args.experiment_name.split(" "))
    save_dir = increment_path(os.path.join(model_dir, args.experiment_name))
    print(f"Model saved to {save_dir}")

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskSplitByProfileDataset
    dataset = dataset_module(data_dir=data_dir,)
    num_classes = dataset.num_classes  # 3 + 2 + 3

    # -- augmentation
    transform_module = getattr(
        import_module("dataset"), args.augmentation
    )  # default: BaseAugmentation
    transform = transform_module(resize=args.resize, crop_size=args.crop_size, mean=dataset.mean, std=dataset.std,)
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(num_classes=num_classes).to(device)
    # torchsummary
    # summary(model, (3, 384, 512))
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion1 = create_criterion(args.criterion1)  # default: cross_entropy
    criterion2 = create_criterion(args.criterion2) # label_smoothing
    criterion3 = create_criterion(args.criterion3) # focal
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4
    )

    # -- scheduler
    if args.scheduler == 'StepLR':
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, args.epochs)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc, best_val_f1, best_val_loss = 0, 0, np.inf

    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        mask_loss_value, gender_loss_value, age_loss_value = 0, 0, 0
        matches = 0
        # mask_matches, gender_matches, age_matches = 0, 0, 0

        model_preds, true_labels = [], []
        mask_preds, true_mask_labels = [], []
        gen_preds, true_gen_labels = [], []
        age_preds, true_age_labels = [], []

        for idx, train_batch in enumerate(train_loader):
            inputs, (mask_labels, gender_labels, age_labels) = train_batch
            inputs = inputs.to(device)

            mask_labels = mask_labels.to(device)
            gender_labels = gender_labels.to(device)
            age_labels = age_labels.to(device)
            labels = mask_labels * 6 + gender_labels * 3 + age_labels
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            (mask_outs, gender_outs, age_outs) = torch.split(outs, [3, 2, 3], dim=1)

            preds_mask = torch.argmax(mask_outs, dim=-1)
            preds_gender = torch.argmax(gender_outs, dim=-1)
            preds_age = torch.argmax(age_outs, dim=-1)
            preds = preds_mask * 6 + preds_gender * 3 + preds_age

            # -- loss
            mask_loss = criterion1(mask_outs, mask_labels)
            gender_loss = criterion2(gender_outs, gender_labels)
            age_loss = criterion3(age_outs, age_labels)

            # weighted loss
            loss_list = [mask_loss, gender_loss, age_loss]
            weight_list = args.loss_rate
            weight_sum = sum(weight_list)
            loss = weighted_loss(loss_list, weight_list)

            loss.backward()
            optimizer.step()
            loss_value += loss.item()
            mask_loss_value += mask_loss.item()
            gender_loss_value += gender_loss.item()
            age_loss_value += age_loss.item()

            matches += (preds == labels).sum().item()
            # mask_matches += (preds_mask == mask_labels).sum().item()
            # gender_matches += (preds_gender == gender_labels).sum().item()
            # age_matches += (preds_age == age_labels).sum().item()
            
            model_preds.extend(preds.detach().cpu().numpy())
            true_labels.extend(labels.detach().cpu().numpy())

            mask_preds.extend(preds_mask.detach().cpu().numpy())
            true_mask_labels.extend(mask_labels.detach().cpu().numpy())

            gen_preds.extend(preds_gender.detach().cpu().numpy())
            true_gen_labels.extend(gender_labels.detach().cpu().numpy())

            age_preds.extend(preds.detach().cpu().numpy())
            true_age_labels.extend(age_labels.detach().cpu().numpy())

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_mask_loss = mask_loss_value / args.log_interval
                train_gender_loss = gender_loss_value / args.log_interval
                train_age_loss = age_loss_value / args.log_interval

                train_acc = matches / args.batch_size / args.log_interval
                # train_mask_acc = mask_matches / args.batch_size / args.log_interval
                # train_gender_acc = gender_matches / args.batch_size / args.log_interval
                # train_age_acc = age_matches / args.batch_size / args.log_interval

                train_f1 = f1_score(true_labels, model_preds, average='macro')
                train_f1_mask = f1_score(true_mask_labels, mask_preds, average='macro')
                train_f1_gen = f1_score(true_gen_labels, gen_preds, average='macro')
                train_f1_age = f1_score(true_age_labels, age_preds, average='macro')

                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || training f1 {train_f1:4.4} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)

                # wandb
                wandb.log({'Tr Avg Loss': train_loss / weight_sum, 'Tr Avg f1': train_f1, 'Tr mask loss': train_mask_loss, 'Tr mask f1': train_f1_mask, 
                'Tr gen loss': train_gender_loss, 'Tr gen f1': train_f1_gen, 'Tr age loss': train_age_loss, 'Tr age f1': train_f1_age})

                loss_value = 0
                mask_loss_value, gender_loss_value, age_loss_value = 0, 0, 0
                matches = 0
                # mask_matches, gender_matches, age_matches = 0, 0, 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_mask_loss_items, val_gender_loss_items, val_age_loss_items = [], [], []
            val_acc_items = []
            figure = None

            model_preds, true_labels = [], []
            mask_preds, true_mask_labels = [], []
            gen_preds, true_gen_labels = [], []
            age_preds, true_age_labels = [], []

            for val_batch in val_loader:
                inputs, (mask_labels, gender_labels, age_labels) = val_batch
                inputs = inputs.to(device)

                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.to(device)
                age_labels = age_labels.to(device)
                labels = mask_labels * 6 + gender_labels * 3 + age_labels
                labels = labels.to(device)

                outs = model(inputs)
                (mask_outs, gender_outs, age_outs) = torch.split(outs, [3, 2, 3], dim=1)

                preds_mask = torch.argmax(mask_outs, dim=-1)
                preds_gender = torch.argmax(gender_outs, dim=-1)
                preds_age = torch.argmax(age_outs, dim=-1)
                preds = preds_mask * 6 + preds_gender * 3 + preds_age

                mask_loss = criterion1(mask_outs, mask_labels)
                gender_loss = criterion2(gender_outs, gender_labels)
                age_loss = criterion3(age_outs, age_labels)

                # weighted loss
                loss_list = [mask_loss, gender_loss, age_loss]
                weight_list = args.loss_rate
                loss = weighted_loss(loss_list, weight_list)

                loss_item = loss.item()
                mask_loss_item, gender_loss_item, age_loss_item = mask_loss.item(), gender_loss.item(), age_loss.item()
                val_loss_items.append(loss_item)
                val_mask_loss_items.append(mask_loss_item)
                val_gender_loss_items.append(gender_loss_item)
                val_age_loss_items.append(age_loss_item)

                matches = (preds == labels).sum().item()
                val_acc_items.append(matches)

                model_preds.extend(preds.detach().cpu().numpy())
                true_labels.extend(labels.detach().cpu().numpy())

                mask_preds.extend(preds_mask.detach().cpu().numpy())
                true_mask_labels.extend(mask_labels.detach().cpu().numpy())

                gen_preds.extend(preds_gender.detach().cpu().numpy())
                true_gen_labels.extend(gender_labels.detach().cpu().numpy())

                age_preds.extend(preds.detach().cpu().numpy())
                true_age_labels.extend(age_labels.detach().cpu().numpy())

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_mask_loss = np.sum(val_mask_loss_items) / len(val_loader)
            val_gender_loss = np.sum(val_gender_loss_items) / len(val_loader)
            val_age_loss = np.sum(val_age_loss_items) / len(val_loader)

            val_acc = np.sum(val_acc_items) / len(val_set)

            val_f1 = f1_score(true_labels, model_preds, average='macro')
            val_f1_mask = f1_score(true_mask_labels, mask_preds, average='macro')
            val_f1_gender = f1_score(true_gen_labels, gen_preds, average='macro')
            val_f1_age = f1_score(true_age_labels, age_preds, average='macro')

            best_val_acc = max(best_val_acc, val_acc)
            best_val_f1 = max(best_val_f1, val_f1)
            if val_loss < best_val_loss:
                print(f"New best model for val loss : {val_loss:4.2}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_loss = val_loss

            print(
                f"[Val] acc : {val_acc:4.2%} || loss : {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%} || best loss : {best_val_loss:4.2} || "
                f"f1 score : {val_f1:4.2} || best f1 : {best_val_f1:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            # logger.add_scalar("Val/f1score", val_f1, epoch)
            # logger.add_figure("results", figure, epoch)
            print()

            # wandb
            wandb.log({'Val Avg Loss': val_loss / weight_sum, 'Val Avg f1': val_f1, 'Val mask loss': val_mask_loss, 'Val mask f1': val_f1_mask, 
                'Val gen loss': val_gender_loss, 'Val gen f1': val_f1_gender, 'Val age loss': val_age_loss, 'Val age f1': val_f1_age})


if __name__ == "__main__":

    CONFIG_FILE_NAME = "./config/config.yaml"
    with open(CONFIG_FILE_NAME, "r") as yml_config_file:
        args = yaml.load(yml_config_file, Loader=yaml.FullLoader)
        args = EasyDict(args["train"])
    
    with open(f'./log/config_{datetime.today()}.yaml', 'w') as f:
        yaml.dump(args, f, default_flow_style=False)
    print(args)
    

    data_dir = args.data_dir
    model_dir = args.model_dir
    CFG = {
        "epochs" : args.epochs,
        "batch_size" : args.batch_size,
        "learning_rate" : args.lr,
        "seed" : args.seed,
        "model" : args.model,
        "optimizer" : args.optimizer,
        "scheduler" : args.scheduler,
        "criterion1" : args.criterion1,
        "criterion2" : args.criterion2,
        "criterion3" : args.criterion3,
        "loss_rate" : args.loss_rate,  
        "img_size" : args.resize,
        "crop_size" : args.crop_size,
        "augmentation" : args.augmentation
    }

    wandb.init(
        project=args.project, entity=args.entity, name=args.experiment_name, config=CFG,
    )

    wandb.define_metric("Train Avg loss", summary="min")
    wandb.define_metric('Tr Avg f1', summary='max')
    wandb.define_metric('Tr mask f1', summary='max')
    wandb.define_metric('Tr gen f1', summary='max')
    wandb.define_metric('Tr age f1', summary='max')

    wandb.define_metric("Val Avg loss", summary="min")
    wandb.define_metric("Val Avg f1", summary="max")
    wandb.define_metric('Val mask f1', summary='max')
    wandb.define_metric('Val gen f1', summary='max')
    wandb.define_metric('Val age f1', summary='max')

    train(data_dir, model_dir, args)

    wandb.finish()