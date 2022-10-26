from easydict import EasyDict
from Dataset.MaskDataset import MaskDataset
from Models.EfficientNet_b0 import EfficientNet_b0
from Models.EfficientNet_b1 import EfficientNet_b1
from Models.EfficientNet_b2 import EfficientNet_b2
import torch
from train import train, validation
from torch.utils.data import DataLoader
from torchvision import transforms
from logger.wandb import logging, finish, init_wandb
import os
import time
from utils.model_manage import save

config = EasyDict({
    'image_root_path': "/opt/ml/input/data/train/images",
    'data_csv_path': '/opt/ml/input/data/train/train_added_label.csv',
    'epochs': 10,
    'batch_size': 64,
    'Train_type': ("Train", "Validation", "Test"),
    'split_rate': 0.8,
    'seed': 444,
    'learning_rate': 1e-3,
    'image_size': (512, 384),
    'desc': 'Standard_b1',
    'is_soft_label': False
})

## mask data standard: [0.558, 0.512, 0.478], [0.218, 0.238, 0.252]
## imagenet standard: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])  

if __name__ == "__main__":
    train_dataset = MaskDataset(
        config.image_root_path,
        config.data_csv_path, 
        config.split_rate, 
        config.Train_type[0],
        False,
        config.seed,
        transforms,
        config.is_soft_label
        )

    val_dataset = MaskDataset(
        config.image_root_path,
        config.data_csv_path, 
        config.split_rate, 
        config.Train_type[1],
        False,
        config.seed,
        transforms,
        config.is_soft_label
        )

    train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_dataloader = DataLoader(val_dataset, config.batch_size, shuffle=True, num_workers=4, drop_last=False)

    model = EfficientNet_b1()
    config['model_name'] = model.name
    init_wandb(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=config.learning_rate*0.1, mode='min')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    criterion.to(device)
    min_val_loss = 1000000

    EXPERIMENT_DIR = f"./runs/{time.strftime('%Y-%m-%d-%H%M%S')}-{config.model_name}-{config.desc}"
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    for epoch in range(config.epochs):
        train_avg_loss, train_score = train(model, optimizer, criterion, train_dataloader, device, epoch)
        val_avg_loss, val_score = validation(model, criterion, val_dataloader, device, epoch)

        if val_avg_loss < min_val_loss:
            save(model, optimizer, EXPERIMENT_DIR)
            print(f"{min_val_loss} -> {val_avg_loss} decreased validation loss -> saved model-{epoch}")
            min_val_loss = val_avg_loss

        scheduler.step(val_avg_loss)

        logging(train_score, train_avg_loss, val_score, val_avg_loss)
    
    finish()