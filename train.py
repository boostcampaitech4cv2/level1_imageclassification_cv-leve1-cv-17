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
from tqdm import tqdm_notebook, notebook
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
# from torch.utils.tensorboard import SummaryWriter
import wandb

from utils.boards import wandb_init
from Dataset.dataset import MaskTestDataset, MaskTrainDataset
from Dataset.data_augmentation import train_transform
from Models.model import EfficientnetB0, EfficientnetB1, EfficientnetB2
from Models.loss import LabelSmoothingCrossEntropy, FocalLoss
from Models.metric import EarlyStopping

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

# pseudo labeling
# 1. Train the model with the training set.
# 2. Forward the test set to the trained model and get the pseudo label.
# 3. Calculate the unlabeled loss using pseudo label.
# 4. train 1 epoch on labeled data(train set) each 50 unlabeled batches.
# 5. repeat 2~4 until the number of epochs is reached.  


def semisup_train(model, 
                  label_train_loader, 
                  unlabel_train_loader, 
                  label_val_loader,
                  criterion, 
                  optimizer, 
                  device, 
                  scheduler):
    alpha = 0
    alpha_t = 1e-4
    T1 = 10
    T2 = 30
    
    EPOCHS = 100
    step = 100
    
    seed_everything(41)
    model.to(device)
    
    best_score = 0
    best_model = None
    early_stopping = EarlyStopping(patience=5, verbose=True, path = f'{os.getcwd()}/Models/saved_model/model_{model.name}_{best_score}_{datetime.today()}.pth')
    
    for epoch in range(EPOCHS):
        
        train_loss = []
        correct = 0
        total = 0
    
        for label_data, unlabel_data in tqdm(zip(iter(label_train_loader), iter(unlabel_train_loader))):
            
            label_img, labels = label_data[0].float().to(device), label_data[1].type(torch.LongTensor).to(device)
            unlabel_img = unlabel_data[0].float().to(device)
            
            optimizer.zero_grad()
            model_pred = model(label_img)
            
            if alpha > 0:
                unlabel_model_pred = model(unlabel_img)
                _, pseudo_labels = torch.max(unlabel_model_pred, dim=1)
                loss = criterion(model_pred, labels) + alpha * criterion(unlabel_model_pred, pseudo_labels)
            
            else:
                loss = criterion(model_pred, labels)
            
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(model_pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).type(torch.float).sum().item()
            
            train_loss.append(loss.item())
            
        if (epoch > T1) and (epoch < T2):
            alpha = alpha_t * ( (epoch - T1) / (T2 - T1) )
        
        elif epoch >= T2:
            alpha = alpha_t
           
        if scheduler is not None:
            scheduler.step()
                
        train_score = correct / total
        tr_loss_mean = np.mean(train_loss)
            
        val_loss, val_score = validation(model, criterion, label_val_loader, device)
            
        print(f'Epoch: {epoch+1}/{EPOCHS} | Step: {step} | Train Loss: {tr_loss_mean:.4f} | Train_score {train_score:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_score:.4f}')
        wandb.log({'train_loss': tr_loss_mean, 'train_score': train_score, 'val_loss': val_loss, 'val_score': val_score})
             
        if best_score < val_score:
            best_model = model
            best_score = val_score

        gc.collect()
        torch.cuda.empty_cache()
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("early stopping")
            break
    
    torch.save(best_model.state_dict(), f'{os.getcwd()}/Models/saved_model/{model.__class__.__name__}_best_{datetime.today()}.pth')
            
    return best_model, best_score        
              


def train(model, optimizer, train_loader, val_loader, criterion, scheduler, device):
    seed_everything(41) # seed = 41 or 444? 
    
    # -- settings
    model.to(device)
    
    best_score = 0
    best_model = None
    EPOCHS = 100
    
    early_stopping = EarlyStopping(patience=5, verbose=True, path = f'{os.getcwd()}/Models/saved_model/model_{model.name}_{best_score}_{datetime.today()}.pth')
    
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
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("early stopping")
            break
        
        gc.collect()
    
    torch.save(best_model.state_dict(), f'{os.getcwd()}/Models/saved_model/model_{model.name}_{best_score}_{datetime.today()}.pth')
    
    return best_model, best_score


    
def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")

def validation(model, criterion, val_loader, device):
    model.eval()
    
    model_preds = []
    true_labels = []
    
    val_loss = []
    
    with torch.no_grad():
        for img, label in tqdm(iter(val_loader)):
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
    unlabel_data_df = pd.read_csv('Dataset/unlabel_data_df.csv')
    
    ## dataset for labeled data
    label_dataset = MaskTrainDataset(data_df, train_transform)   
    label_train_dataset, label_val_dataset = label_dataset.split_dataset()
    
    label_train_loader = DataLoader(label_train_dataset, batch_size=32, shuffle=True, num_workers=4)
    label_val_loader = DataLoader(label_val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    
    ## dataset for unlabeled data 
    unlabel_train_dataset = MaskTrainDataset(unlabel_data_df, train_transform)
    unlabel_train_loader = DataLoader(unlabel_train_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    config = {
        "learning_rate": 0.001,
        "epochs": 20,
        "batch_size": 32,
        "seed": 41,
        "img_size": (220, 224),
        "optimizer": "AdamW",
        "loss": "FocalLoss - gamma=2, with class_weight",
        "scheduler": "CosineAnnealingLR",
        "model": "EfficientnetB1",
        "split": "0.2"
    }
    wandb_init(config)
    
    model = EfficientnetB2()
    
    # print(model.name)

    criterion = FocalLoss(alpha=None, gamma=2).to(device)    
    optimizer = AdamW(model.parameters(), lr=0.001)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True, eps=1e-08)

    
    gc.collect()
    torch.cuda.empty_cache()
    infer_model, infer_score = train(model, optimizer, label_train_loader, label_val_loader, criterion, scheduler, device)
    # pseudo_model, pseudo_score = semisup_train(model, label_train_loader, unlabel_train_loader, label_val_loader, criterion, optimizer, device, scheduler)
    print(infer_score)
    # print(pseudo_score) 
   