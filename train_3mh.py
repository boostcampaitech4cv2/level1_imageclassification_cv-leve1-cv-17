from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import torch

def train(model, optimizer, criterions, dataloader, device, epoch, loss_rates):
    model.train()
    mask_predicts, mask_label_list, mask_loss_list = [], [], []
    gender_predicts, gender_label_list, gender_loss_list = [], [], []
    age_predicts, age_label_list, age_loss_list = [], [], []
    train_loss = []

    for images, mask_labels, gender_labels, age_labels in tqdm(dataloader):
        images, mask_labels, gender_labels, age_labels = images.to(device), mask_labels.to(device), gender_labels.float().to(device), age_labels.to(device)

        optimizer.zero_grad()
        mask_predict, gender_predict, age_predict = model(images)
        mask_predict, gender_predict, age_predict = mask_predict.view(images.size(0), -1), gender_predict.view(images.size(0), -1), age_predict.view(images.size(0), -1)

        mask_loss = criterions[0](mask_predict, mask_labels)*loss_rates[0]
        gender_loss = criterions[1](gender_predict, gender_labels)*loss_rates[1]
        age_loss = criterions[2](age_predict, age_labels)*loss_rates[2]

        loss = (mask_loss + gender_loss + age_loss) / len(criterions)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        mask_loss_list.append(mask_loss.item())
        gender_loss_list.append(gender_loss.item())
        age_loss_list.append(age_loss.item())

        mask_predicts += mask_predict.argmax(1).cpu().numpy().tolist()
        gender_predicts += gender_predict.argmax(1).cpu().numpy().tolist()
        age_predicts += age_predict.argmax(1).cpu().numpy().tolist()

        mask_label_list += mask_labels.cpu().numpy().tolist()
        gender_label_list += gender_labels.argmax(1).cpu().numpy().tolist()
        age_label_list += age_labels.cpu().numpy().tolist()
    
    train_avg_loss = np.mean(train_loss)
    mask_avg_loss = np.mean(mask_loss_list)
    gender_avg_loss = np.mean(gender_loss_list)
    age_avg_loss = np.mean(age_loss_list)

    mask_score = f1_score(mask_label_list, mask_predicts, average="macro")
    gender_score = f1_score(gender_label_list, gender_predicts, average="macro")
    age_score = f1_score(age_label_list, age_predicts, average="macro")

    print(f"Train Epoch: {epoch} loss: {train_avg_loss} mask loss: {mask_avg_loss} age loss: {age_avg_loss} gender loss: {gender_avg_loss} mask score: {mask_score} age score: {age_score} gender score: {gender_score}")
    
    return (train_avg_loss, mask_avg_loss, gender_avg_loss, age_avg_loss, mask_score, gender_score, age_score)

def validation(model, criterions, dataloader, device, epoch, loss_rates):
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

        print(f"Validation Epoch: {epoch} loss: {val_avg_loss} mask loss: {mask_avg_loss} age loss: {age_avg_loss} gender loss: {gender_avg_loss} mask score: {mask_score} age score: {age_score} gender score: {gender_score}")
            
    return (val_avg_loss, mask_avg_loss, gender_avg_loss, age_avg_loss, mask_score, gender_score, age_score)