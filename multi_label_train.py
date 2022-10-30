import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

import wandb

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
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

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


def train(data_dir, model_dir, args):
    wandb.login

    wandb.init(project = "mask_classification", entity = "cv17",
                config = {"batch_size": args.batch_size,
                        "learning_rate" : args.lr,
                        "epochs"    : args.epochs,
                        "model_name"  : args.model,
                        "optimizer" : args.optimizer,
                        "seed"  : args.seed
                })


    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.train_augmentation)  # CustomAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    
    dataset.set_transform(transform)
    train_set, val_set = dataset.split_dataset()

    # -- data_loader
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model) 
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # cross_entropy
    criterion2 = create_criterion(args.criterion2) # label_smoothing
    criterion3 = create_criterion(args.criterion3) # focal
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: AdamW
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-2
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_loss = np.inf
    best_val_acc = 0

    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        mask_matches, gender_matches, age_matches = 0, 0, 0

        for idx, train_batch in enumerate(train_loader):
            inputs, (mask_labels, gender_labels, age_labels) = train_batch
            inputs = inputs.to(device)

            labels = torch.stack((mask_labels, gender_labels, age_labels), dim=1)
            mask_labels, gender_labels, age_labels = mask_labels.to(device), gender_labels.to(device), age_labels.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            
            (mask_outs, gender_outs, age_outs) = torch.split(outs, [3, 2, 3], dim=1)
            
            preds_mask = torch.argmax(mask_outs, dim=-1) 
            preds_gender = torch.argmax(gender_outs, dim=-1) 
            preds_age = torch.argmax(age_outs, dim=-1) 
            preds = torch.stack((preds_mask, preds_gender, preds_age), dim=1)

            mask_loss = criterion(mask_outs, mask_labels) # crossentropy
            gender_loss = criterion2(gender_outs, gender_labels) # label_smoothing
            age_loss = criterion3(age_outs, age_labels) # focal

            loss = mask_loss + gender_loss + 1.5 * age_loss
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += torch.all((preds==labels), dim=1).sum().item()
            mask_matches += (preds_mask == mask_labels).sum().item()
            gender_matches += (preds_gender == gender_labels).sum().item()
            age_matches += (preds_age == age_labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval 
                train_acc = matches / args.batch_size / args.log_interval
                train_mask_acc = mask_matches / args.batch_size / args.log_interval
                train_gender_acc = gender_matches / args.batch_size / args.log_interval
                train_age_acc = age_matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch + 1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || acc {train_acc:4.2%} || mask acc {train_mask_acc:4.2%} || gen acc {train_gender_acc:4.2%} || age acc {train_age_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                wandb.log({'Train Avg Loss': train_loss, 'Train Acc': train_acc, 'Mask Acc': train_mask_acc, 'Gen Acc': train_gender_acc, 'Age Acc': train_age_acc})
                
                loss_value = 0
                matches = 0
                mask_matches, gender_matches, age_matches = 0, 0, 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, (mask_labels, gender_labels, age_labels) = val_batch

                inputs = inputs.to(device)
                labels = torch.stack((mask_labels, gender_labels, age_labels), dim=1)
                mask_labels, gender_labels, age_labels = mask_labels.to(device), gender_labels.to(device), age_labels.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                (mask_outs, gender_outs, age_outs) = torch.split(outs, [3, 2, 3], dim=1)

                preds_mask = torch.argmax(mask_outs, dim=-1) 
                preds_gender = torch.argmax(gender_outs, dim=-1) 
                preds_age = torch.argmax(age_outs, dim=-1)
                preds = torch.stack((preds_mask, preds_gender, preds_age), dim=1)

                mask_loss = criterion(mask_outs, mask_labels) # crossentropy , 기존의 loss
                gender_loss = criterion2(gender_outs, gender_labels) # label_smoothing
                age_loss = criterion3(age_outs, age_labels) # focal

                loss = mask_loss + gender_loss + 1.5 * age_loss
                loss_item = loss.item()

                matches = torch.all((preds==labels), dim=1).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(matches)
            
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            # best_val_acc = max(val_acc, best_val_acc)
            best_val_loss = min(val_loss, best_val_loss)
            if val_acc > best_val_acc:   
                print(f"New best model for val acc : {val_acc:4.2}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                # best_val_loss = val_loss
                best_val_acc = val_acc
            print(f"[Val] acc : {val_acc:4.2%} || loss: {val_loss:4.2} || best vac : {best_val_acc:4.2%} || best loss: {best_val_loss:4.4}")
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            print()
            wandb.log({
                    "Validation Avg Loss": val_loss,
                    "Validation Accuracy" : val_acc
                    })

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 444)')
    parser.add_argument('--epochs', type=int, default=13, help='number of epochs to train (default: 20)')
    parser.add_argument('--dataset', type=str, default='MaskMultiLabelDataset', help='dataset augmentation type (default: MaskMultiLabelDataset)')
    parser.add_argument('--train_augmentation', type=str, default='MyAugmentation', help='data augmentation type')
    parser.add_argument('--val_augmentation', type=str, default='BaseAugmentation', help='data augmentation type')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='ResNet34', help='model type (default: EfficientNet_B0)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--criterion2', type=str, default='cross_entropy', help='criterion type (default: label_smoothing)')
    parser.add_argument('--criterion3', type=str, default='focal', help='criterion type (default: focal)')
    parser.add_argument('--scheduler', type=str, default='StepLR')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
