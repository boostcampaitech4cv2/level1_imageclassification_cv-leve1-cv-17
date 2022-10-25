import os
import numpy as np
import torch
import cv2
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
from enum import Enum

from face_detections import cv_face_detect, face_detect, gray_scale
    

# get image path from input/data/train/train.csv
def get_image_path(data_dir, train=True):
    if train:
        image_path = []
        image_label = []
        df = pd.read_csv(os.path.join(data_dir, "input", "data", "train", "train.csv"))
        for i in range(len(df)):
            image_path.append(os.path.join(data_dir, "input", "data" "train", "images", df.iloc[i, -1]))
            image_label.append(df.iloc[i, -1])
        return image_path, image_label

    # get image path from input/data/eval/images
    # check it
    '''
    else:
        image_path = []
        for i in os.listdir(os.path.join(data_dir, 'input', 'data', 'eval', 'images')):
            image_path.append(os.path.join(data_dir, 'input', 'data', 'eval', 'images', i))
        return image_path
    '''
def mask_label(img_path, img_label):
    mask = 0
    gender = 0
    age = 0
    
    img_info = img_label.split('_')
    print(img_info)
    # mask check
    for i in os.path.join(img_path):
        if 'mask' in i:
            mask = 0
        elif 'incorrect_mask' in i:
            mask = 1
        else:
            mask = 2
    
    # gender check
    if img_info[1] == 'male':
        gender = 0
    else:
        gender = 1
    
    # age check
    if int(img_info[3]) < 30:
        age = 0
    elif 30 <= int(img_info[3]) < 60:
        age = 1
    else:
        age = 2
        
    return 6*mask + 3*gender + age

    
class MaskDataset(Dataset):
    
    num_classes = 3 * 2 * 3
    def __init__(self, image_path, transform=None):
        super(Dataset, self).__init__()
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # self.face = face_detect(self.image, gray_scale)
        self.transform = transform
    
    def __getitem__(self,index):
        return (self.data[index], self.label[index])
    
    def __len__(self):
        return len(self.data)
    
    
if __name__ == '__main__':
    # os.getcwd() = /opt/ml/project/
    df = pd.read_csv(os.path.join(os.getcwd(), "input", "data", "train", "train.csv"))
    print(df.iloc[0, -1])
    print(os.path.join(os.getcwd(), "input", "data", "train", "train.csv"))
    image_path, image_label = get_image_path(os.getcwd())
    print(image_path[0])
    print(image_label[0])
    print(mask_label(image_path[0], image_label[0]))