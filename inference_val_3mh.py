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
    gender_predicts, gender_label_list, gender_loss_list = [], [], []
    age_predicts, age_label_list, age_loss_list = [], [], []
    val_loss = []

    with torch.no_grad():
        for images, mask_labels, gender_labels, age_labels in tqdm(dataloader):
            images, mask_labels, gender_labels, age_labels = images.to(device), mask_labels.to(device), gender_labels.float().to(device), age_labels.to(device)

            mask_predict, gender_predict, age_predict = model(images)
            mask_predict, gender_predict, age_predict = mask_predict.view(images.size(0), -1), gender_predict.view(images.size(0), -1), age_predict.view(images.size(0), -1)

            mask_loss = criterions[0](mask_predict, mask_labels)*loss_rates[0]
            gender_loss = criterions[1](gender_predict, gender_labels)*loss_rates[1]
            age_loss = criterions[2](age_predict, age_labels)*loss_rates[2]

            loss = (mask_loss + gender_loss + age_loss) / len(criterions)

            val_loss.append(loss.item())
            mask_loss_list.append(mask_loss.item())
            gender_loss_list.append(gender_loss.item())
            age_loss_list.append(age_loss.item())

            mask_predicts += mask_predict.argmax(1).detach().cpu().numpy().tolist()
            gender_predicts += gender_predict.argmax(1).detach().cpu().numpy().tolist()
            age_predicts += age_predict.argmax(1).detach().cpu().numpy().tolist()

            mask_label_list += mask_labels.detach().cpu().numpy().tolist()
            gender_label_list += gender_labels.argmax(1).detach().cpu().numpy().tolist()
            age_label_list += age_labels.detach().cpu().numpy().tolist()
        
        val_avg_loss = np.mean(val_loss)
        mask_avg_loss = np.mean(mask_loss_list)
        gender_avg_loss = np.mean(gender_loss_list)
        age_avg_loss = np.mean(age_loss_list)

        mask_score = f1_score(mask_label_list, mask_predicts, average="macro")
        gender_score = f1_score(gender_label_list, gender_predicts, average="macro")
        age_score = f1_score(age_label_list, age_predicts, average="macro")

        print(f"Inference Validation loss: {val_avg_loss} mask loss: {mask_avg_loss} age loss: {age_avg_loss} gender loss: {gender_avg_loss} mask score: {mask_score} age score: {age_score} gender score: {gender_score}")

    labels = [i for i in range(18)]
    predicts, val_labels = compose_label(mask_predicts, gender_predicts, age_predicts, mask_label_list, gender_label_list, age_label_list)

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
        file.write(f'val_gender_loss: {gender_avg_loss}')
        file.write(f'val_age_loss: {age_avg_loss}')
        file.write(f'val_mask_score: {mask_score}')
        file.write(f'val_gender_score: {gender_score}')
        file.write(f'val_age_score: {age_score}')
        file.close()

def compose_label(mask_predicts, gender_predicts, age_predicts, mask_label_list, gender_label_list, age_label_list):
    label_init_list, val_labels, predicts = [0, 6, 12], [], []

    for mask_predict, gender_predict, age_predict, mask_label, gender_label, age_label in zip(mask_predicts, gender_predicts, age_predicts, mask_label_list, gender_label_list, age_label_list):
        predict_label = mask_predict*6 + gender_predict*3 + age_predict
        label = mask_label*6 + gender_label*3 + age_label

        predicts.append(predict_label)
        val_labels.append(label)

    return predicts, val_labels