import random
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
# from torch.utils.tensorboard import SummaryWriter

from Dataset.dataset import MaskTestDataset, MaskTrainDataset
from Dataset.data_augmentation import train_transform
from Models.model import EfficientnetB0

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    
def train(model, optimizer, train_loader, val_loader, criterion, scheduler, device):
    seed_everything(41) # seed = 41
    
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
        
        if scheduler is not None:
            scheduler.step()
        
        if best_score < val_score:
            best_model = model
            best_score = val_score
            
        gc.collect()
    
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
    
    model = EfficientnetB0()
    model.eval()
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00001)
    
    gc.collect()
    torch.cuda.empty_cache()
    infer_model, infer_score = train(model, optimizer, train_loader, val_loader, criterion, scheduler, device)
    
    print(infer_score)
    