from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import torch

def train(model, optimizer, criterion, dataloader, device, epoch):
    model.train()
    predicts, train_labels, train_loss = [], [], []

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.long().to(device)

        optimizer.zero_grad()
        predict = model(images)
        loss = criterion(predict, labels)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        predicts += predict.argmax(1).cpu().numpy().tolist()
        train_labels += labels.cpu().numpy().tolist()
    
    train_avg_loss = np.mean(train_loss)
    train_score = f1_score(train_labels, predicts, average="macro")

    print(f"Train Epoch: {epoch} loss: {train_avg_loss} F1-score: {train_score}")
    
    return (train_avg_loss, train_score)


def validation(model, criterion, dataloader, device, epoch):
    model.eval()
    predicts, val_labels, val_loss = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.long().to(device)

            predict = model(images)
            loss = criterion(predict, labels).detach()
    
            val_loss.append(loss.item())
            predicts += predict.argmax(1).detach().cpu().numpy().tolist()
            val_labels += labels.detach().cpu().numpy().tolist()
            
        val_avg_loss = np.mean(val_loss)
        val_score = f1_score(val_labels, predicts, average="macro")

        print(f"Validation epoch: {epoch} loss: {val_avg_loss} F1-score: {val_score}")

    return (val_avg_loss, val_score)