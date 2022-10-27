from utils.model_manage import load_checkpoint
import torch
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def inference_val(checkpoint, model, optimizer, criterion, dataloader, device):
    load_checkpoint(torch.load(f'{checkpoint}/checkpoint.pth.tar'), model, optimizer)

    model.eval()
    predicts, val_labels = [], []
    val_loss = []

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

        print(f"Inference Validation loss: {val_avg_loss} F1-score: {val_score}")

    labels = [i for i in range(18)]

    cm = confusion_matrix(val_labels, predicts, labels=labels)

    df_cm = pd.DataFrame(cm)
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel('Model Predict')
    plt.ylabel('Ground Truth')
    plt.savefig(checkpoint + '/confusion_matrix.png')

    with open(f'{checkpoint}/inference_val.txt', "w") as file:
        file.write(f'val_score: {val_score}\n')
        file.write(f'val_avg_loss: {val_avg_loss}')
        file.close()