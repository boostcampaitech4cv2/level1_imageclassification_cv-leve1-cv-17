from utils.model_manage import load_checkpoint
import torch
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def inference_val(checkpoint, model, optimizer, criterions, dataloader, device, loss_rates):
    load_checkpoint(torch.load(f'{checkpoint}/checkpoint.pth.tar'), model, optimizer)

    model.eval()
    mask_predicts, mask_label_list, mask_loss_list = [], [], []
    age_gender_predicts, age_gender_label_list, age_gender_loss_list = [], [], []
    val_loss = []

    with torch.no_grad():
        for images, mask_labels, age_gender_labels in tqdm(dataloader):
            images, mask_labels, age_gender_labels = images.to(device), mask_labels.long().to(device), age_gender_labels.long().to(device)

            mask_predict, age_gender_predict = model(images)

            mask_loss = criterions[0](mask_predict, mask_labels)*loss_rates[0]
            age_gender_loss = criterions[1](age_gender_predict, age_gender_labels)*loss_rates[1]

            loss = (mask_loss + age_gender_loss) / len(criterions)

            val_loss.append(loss.item())
            mask_loss_list.append(mask_loss.item())
            age_gender_loss_list.append(age_gender_loss.item())

            mask_predicts += mask_predict.argmax(1).detach().cpu().numpy().tolist()
            age_gender_predicts += age_gender_predict.argmax(1).detach().cpu().numpy().tolist()

            mask_label_list += mask_labels.detach().cpu().numpy().tolist()
            age_gender_label_list += age_gender_labels.detach().cpu().numpy().tolist()
        
        val_avg_loss = np.mean(val_loss)
        mask_avg_loss = np.mean(mask_loss_list)
        age_gender_avg_loss = np.mean(age_gender_loss_list)

        mask_score = f1_score(mask_label_list, mask_predicts, average="macro")
        age_gender_score = f1_score(age_gender_label_list, age_gender_predicts, average="macro")

        print(f"Inference Validation loss: {val_avg_loss} mask loss: {mask_avg_loss} ag loss: {age_gender_avg_loss} mask score: {mask_score} ag score: {age_gender_score}")

    labels = [i for i in range(18)]
    predicts, val_labels = compose_label(mask_predicts, age_gender_predicts, mask_label_list, age_gender_label_list)

    cm = confusion_matrix(val_labels, predicts, labels=labels)
    df_cm = pd.DataFrame(cm)
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel('Model Predict')
    plt.ylabel('Ground Truth')
    plt.savefig(checkpoint + '/confusion_matrix.png')

    with open(f'{checkpoint}/inference_val.txt', "w") as file:
        file.write(f'val_avg_loss: {val_avg_loss}')
        file.write(f'val_mask_loss: {mask_avg_loss}')
        file.write(f'val_age_gender_loss: {age_gender_avg_loss}')
        file.write(f'val_mask_score: {mask_score}')
        file.write(f'val_age_gender_score: {age_gender_score}')
        file.close()

def compose_label(mask_predicts, age_gender_predicts, mask_label_list, age_gender_label_list):
    label_init_list, val_labels, predicts = [0, 6, 12], [], []

    for mask_predict, age_gender_predict, mask_label, age_gender_label in zip(mask_predicts, age_gender_predicts, mask_label_list, age_gender_label_list):
        predict_label, label = 0, 0

        if mask_predict == 0:
            predict_label = label_init_list[0] + age_gender_predict
        elif mask_predict == 1:
            predict_label = label_init_list[1] + age_gender_predict
        elif mask_predict == 2:
            predict_label = label_init_list[2] + age_gender_predict

        if mask_label == 0:
            label = label_init_list[0] + age_gender_label
        elif mask_label == 1:
            label = label_init_list[1] + age_gender_label
        elif mask_label == 2:
            label = label_init_list[2] + age_gender_label

        predicts.append(predict_label)
        val_labels.append(label)

    return predicts, val_labels