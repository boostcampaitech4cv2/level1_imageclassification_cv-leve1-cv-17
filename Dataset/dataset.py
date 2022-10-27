import os
from glob import glob
from typing import Tuple
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import random

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset, random_split
from sklearn.model_selection import train_test_split
from enum import Enum

from Dataset.data_augmentation import train_transform, test_transform

# from face_detections import cv_face_detect, face_detect, gray_scale
    
num2class = ['incorrect_mask', 'mask1', 'mask2', 'mask3',
             'mask4', 'mask5', 'normal']

class CFG:
    data_dir = os.getcwd() + '/input/data/train' # /opt/ml/project/input/data/train
    img_dir = f'{data_dir}/images'
    df_path = f'{data_dir}/train.csv'
    
    
def get_ext(img_dir, img_id):
    """
    학습 데이터셋 이미지 폴더에는 여러 하위폴더로 구성되고, 이 하위폴더들에는 각 사람의 사진들이 들어가있습니다. 하위폴더에 속한 이미지의 확장자를 구하는 함수입니다.
    
    Args:
        img_dir: 학습 데이터셋 이미지 폴더 경로 
        img_id: 학습 데이터셋 하위폴더 이름

    Returns:
        ext: 이미지의 확장자
    """
    filename = os.listdir(os.path.join(img_dir, img_id))[0]
    ext = os.path.splitext(filename)[-1].lower()
    return ext

def data_making(label_df, data_feeding):
    index = 0
    try:
        for file_name in label_df['image_path']: # file_name '/opt/ml/project/input/data/train/images/000001_female_Asian_45'
            for image in os.listdir(file_name): # image 'mask3.jpg'
                image_path= os.path.join(file_name, image) # image_path = '/opt/ml/project/input/data/train/images/000001_female_Asian_45/mask3.jpg'
                label = image.split('.')[0] # label = 'mask3'
                label_number = label_df[label_df['image_path'] == file_name][label].values[0] # label_number = 4
                data_feeding.append([image_path, label_number]) # data_feeding = [['/opt/ml/project/input/data/train/images/000001_female_Asian_45/mask3.jpg', 3]]
            index += 1
    except KeyError:
        print(index) # 2557 (error)
        print(label_df['image_path'][index]) #/opt/ml/project/input/data/train/images/006578_male_Asian_19
    
    return data_feeding
    
class MaskTrainDataset(Dataset):
    
    def __init__(self, data_df, transform=None, val_ratio=0.2):
        super(Dataset, self).__init__()
        self.df = data_df
        self.img_dir = self.df['image_path']
        self.transform = transform
        self.img_label = self.df['label']
        self.val_ratio = val_ratio
    
    
    def __getitem__(self, index):
        img_path = self.img_dir[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = img[193:193+195, 102:102+176]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            img = self.transform(image=img)['image']
        
        if self.img_label is not None:
            label = self.img_label[index]
            return img, label
    
    def __len__(self):
        return len(self.img_dir)
    
    def split_dataset(self) -> Tuple[Subset, Subset]:
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set
        

class MaskTestDataset(Dataset):
    def __init__(self, img_path, transform=None):
        super(Dataset, self).__init__()
        self.df = data_df
        self.img_path = img_path
        self.transform = transform
        
    def __getitem__(self, index):
        image = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return image
    
    def __len__(self):
        return len(self.img_path)
       
if __name__ == '__main__':
    # os.getcwd() = /opt/ml/project/
    df = pd.read_csv(os.path.join(os.getcwd(), "input", "data", "train", "train.csv"))
    label_df = pd.read_csv(os.path.join(os.getcwd(), "input", "data", "train", "label.csv"))
    
    data_feeding = []
    image_folder_path = os.path.join(os.getcwd(), "input", "data", "train", "images") # /opt/ml/project/input/data/train/images
    image_folder = os.listdir(image_folder_path) # image_folder = ['00000', '00001', '00002', ...]
    
    if 'data_df.csv' not in os.listdir(os.path.join(os.getcwd(), 'Dataset')) :
        print('data df making')
        data_making(label_df, data_feeding)
        data_df = pd.DataFrame(data_feeding, columns=['image_path', 'label'])
        print(data_df.head())
        data_df.to_csv(os.path.join(os.getcwd(), "Dataset", "data_df.csv"), index=False)
    else:
        data_df = pd.read_csv(os.path.join(os.getcwd(), "Dataset", "data_df.csv"))
        
    dataset = MaskTrainDataset(data_df, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    # print(list(iter(dataloader))[0][0].shape)
    print(dataset.__getitem__(0))
    
    train_loader, val_loader = dataset.split_dataset() # train_loader, val_loader = Subset, Subset -> model에서 동작될려나? -> 동작됨
    print(len(train_loader), len(val_loader))
    print(next(iter(train_loader))[0].shape)