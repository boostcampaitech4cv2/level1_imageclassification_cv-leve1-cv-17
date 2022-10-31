from easydict import EasyDict
from Dataset.MaskDataset import MaskDataset
from Dataset.MaskDataset_mh import MaskDataset_mh
import torch
from train import train, validation
from torch.utils.data import DataLoader
from torchvision import transforms
from logger.wandb import logging, finish, init_wandb
import os
import time
from utils.model_manage import save
from baseline.loss import create_criterion
from Modules.EfficientNet import create_model
import random
import numpy as np
from inference_val import inference_val
import json

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    config = EasyDict({
        'image_root_path': "/opt/ml/input/data/train/images",
        'data_csv_path': '/opt/ml/input/data/train/train_added_label.csv',
        'epochs': 10,
        'batch_size': 64,
        'Train_type': ("Train", "Validation", "Test"),
        'optimizer': 'Adam',
        'model_name': "EfficientNet_b0",
        'loss': 'focal',  ## cross_entropy, focal
        'split_rate': 0.8,
        'seed': 41,
        'learning_rate': 1e-3,
        'image_size': (512, 384),
        'desc': 'Normalize_FL_b1_BM',
        'is_soft_label': False,
        'mean': [0.548, 0.504, 0.479],  ## mask: [0.558, 0.512, 0.478], imageNet: [0.485, 0.456, 0.406], baseline: [0.548, 0.504, 0.479]
        'std': [0.237, 0.247, 0.246]   ## mask: [0.218, 0.238, 0.252], imageNet: [0.229, 0.224, 0.225], baseline: [0.237, 0.247, 0.246]
    })

    transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)  
    ])  

    seed_everything(config.seed)

    train_dataset = MaskDataset_mh(
        image_root_path = config.image_root_path,
        data_csv_path = config.data_csv_path, 
        split_rate = config.split_rate, 
        train_type = config.Train_type[0],
        is_inference = False,
        transform = transforms,
        is_soft_label = config.is_soft_label
        )

    val_dataset = MaskDataset_mh(
        image_root_path = config.image_root_path,
        data_csv_path = config.data_csv_path, 
        split_rate = config.split_rate, 
        train_type = config.Train_type[1],
        is_inference = False,
        transform = transforms,
        is_soft_label = config.is_soft_label
        )

    train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_dataloader = DataLoader(val_dataset, config.batch_size, shuffle=False, num_workers=4, drop_last=False)

    model = create_model(config.model_name)
    init_wandb(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = create_criterion(config.loss) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=config.learning_rate*0.1, mode='min')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    criterion.to(device)
    min_val_loss = 1000000

    EXPERIMENT_DIR = f"./runs/{time.strftime('%Y-%m-%d-%H%M%S')}-{config.model_name}-{config.desc}"
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    scheduler_step = 0

    for epoch in range(config.epochs):
        train_avg_loss, train_score = train(model, optimizer, criterion, train_dataloader, device, epoch)
        val_avg_loss, val_score = validation(model, criterion, val_dataloader, device, epoch)

        if val_avg_loss < min_val_loss:
            save(model, optimizer, EXPERIMENT_DIR)
            print(f"{min_val_loss} -> {val_avg_loss} decreased validation loss -> saved model-{epoch}")
            min_val_loss = val_avg_loss
            scheduler_step = 0
        else:
            scheduler_step += 1

        scheduler.step(val_avg_loss)

        logging(train_score, train_avg_loss, val_score, val_avg_loss)
        
        if scheduler_step > 3:
            print("decrease learning rate base*0.5")
            scheduler_step = 0
    
    finish()

    inference_val(f'{EXPERIMENT_DIR}', model, optimizer, criterion, val_dataloader, device)

    with open(f'{EXPERIMENT_DIR}/config.json', 'w') as file:
        json.dump(config, file, ensure_ascii=False, indent=4)