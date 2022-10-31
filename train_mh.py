from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import torch

def train(model, optimizer, criterions, dataloader, device, epoch, loss_rates):
    model.train()
    mask_predicts, mask_label_list, mask_loss_list = [], [], []
    age_gender_predicts, age_gender_label_list, age_gender_loss_list = [], [], []
    train_loss = []

    for images, mask_labels, age_gender_labels in tqdm(dataloader):
        images, mask_labels, age_gender_labels = images.to(device), mask_labels.long().to(device), age_gender_labels.long().to(device)

        optimizer.zero_grad()
        mask_predict, age_gender_predict = model(images)
        mask_loss = criterions[0](mask_predict, mask_labels)*loss_rates[0]
        age_gender_loss = criterions[1](age_gender_predict, age_gender_labels)*loss_rates[1]

        loss = (mask_loss + age_gender_loss) / len(criterions)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        mask_loss_list.append(mask_loss.item())
        age_gender_loss_list.append(age_gender_loss.item())

        mask_predicts += mask_predict.argmax(1).cpu().numpy().tolist()
        age_gender_predicts += age_gender_predict.argmax(1).cpu().numpy().tolist()

        mask_label_list += mask_labels.cpu().numpy().tolist()
        age_gender_label_list += age_gender_labels.cpu().numpy().tolist()
    
    train_avg_loss = np.mean(train_loss)
    mask_avg_loss = np.mean(mask_loss_list)
    age_gender_avg_loss = np.mean(age_gender_loss_list)

    mask_score = f1_score(mask_label_list, mask_predicts, average="macro")
    age_gender_score = f1_score(age_gender_label_list, age_gender_predicts, average="macro")

    print(f"Train Epoch: {epoch} loss: {train_avg_loss} mask loss: {mask_avg_loss} ag loss: {age_gender_avg_loss} mask score: {mask_score} ag score: {age_gender_score}")
    
    return (train_avg_loss, mask_avg_loss, age_gender_avg_loss, mask_score, age_gender_score)


def validation(model, criterions, dataloader, device, epoch, loss_rates):
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

        print(f"Validation Epoch: {epoch} loss: {val_avg_loss} mask loss: {mask_avg_loss} ag loss: {age_gender_avg_loss} mask score: {mask_score} ag score: {age_gender_score}")
            
    return (val_avg_loss, mask_avg_loss, age_gender_avg_loss, mask_score, age_gender_score)