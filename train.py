import random
import gc
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
# from torch.utils.tensorboard import SummaryWriter
import wandb

from utils.boards import wandb_init
from Dataset.dataset import MaskTestDataset, MaskTrainDataset
from Dataset.data_augmentation import train_transform
from Models.model import EfficientnetB0, EfficientnetB1, EfficientnetB2
from Models.loss import LabelSmoothingCrossEntropy

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# stratified k-fold cross validation
# get the labels from /opt/ml/project/Dataset/data_df.csv
# def k_fold_cross_validation(model, optimizer, train_loader, val_loader, criterion, scheduler, device, k=5):
#    seed_everything(41)
#    
    
def train(model, optimizer, train_loader, val_loader, criterion, scheduler, device):
    seed_everything(41) # seed = 41 or 444? 
    
    # -- settings
    model.to(device)
    
    best_score = 0
    best_model = None
    
    for epoch in range(1, 21):
        model.train()
        train_loss = []
        train_score = 0
        train_correct = 0
        
        for img, label in tqdm(iter(train_loader)):
            img, label = img.float().to(device), label.type(torch.LongTensor).to(device)
            if img is None:
                print('train img is empty!')
            optimizer.zero_grad()
            
            model_pred = model(img)
            loss = criterion(model_pred, label)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            train_correct += (model_pred.argmax(1) == label).type(torch.float).sum().item()
            
        train_score = train_correct / len(train_loader.dataset)
        
        tr_loss = np.mean(train_loss)
        
        val_loss, val_score = validation(model, criterion, val_loader, device)
        
        print(f'Epoch: {epoch}, Train Loss: {tr_loss:.4f}, Train Score: {train_score:.4f}, Val Loss: {val_loss:.4f}, Val Score: {val_score:.4f}')
        wandb.log({'train_loss': tr_loss, 'train_score': train_score, 'val_loss': val_loss, 'val_score': val_score})
        
        if scheduler is not None:
            # scheduler.step(val_loss) # ReduceLROnPlateau
            scheduler.step() # CosineAnnealingLR
        
        if best_score < val_score:
            best_model = model
            best_score = val_score
            
        
            
        gc.collect()
    
    torch.save(best_model.state_dict(), f'{os.getcwd()}/Models/model_{datetime.today()}.pth')
    
    return best_model, best_score

    
def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")

def validation(model, criterion, test_loader, device):
    model.eval()
    
    model_preds = []
    true_labels = []
    
    val_loss = []
    
    with torch.no_grad():
        for img, label in tqdm(iter(test_loader)):
            img, label = img.float().to(device), label.type(torch.LongTensor).to(device)
            if img is None:
                print('validation img is empty!')
            
            model_pred = model(img)
            
            loss = criterion(model_pred, label)
            
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
        
    val_f1 = competition_metric(true_labels, model_preds)
    return np.mean(val_loss), val_f1    


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_df = pd.read_csv('Dataset/data_df.csv')
    dataset = MaskTrainDataset(data_df, train_transform)
    
    train_dataset, val_dataset = dataset.split_dataset()
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    config = {
        "learning_rate": 0.001,
        "epochs": 20,
        "batch_size": 32,
        "seed": 41,
        "img_size": (220, 224),
        "optimizer": "SGD",
        "loss": "LabelSmoothingCrossEntropy",
        "scheduler": "CosineAnnealingLR",
        "model": "EfficientnetB2",
        "split": "0.2"
    }
    wandb_init(config)
    
    model = EfficientnetB2()
    model.eval()
    
    class_weight = torch.FloatTensor(1 / data_df['label'].value_counts()).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = LabelSmoothingCrossEntropy().to(device)
    
    # optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = AdamW(model.parameters(), lr=0.001)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_max=10, eta_min=0.00001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True, eps=1e-08)
    
    gc.collect()
    torch.cuda.empty_cache()
    infer_model, infer_score = train(model, optimizer, train_loader, val_loader, criterion, scheduler, device)
    
    print(infer_score)
    
    