import os
from glob import glob
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import random

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
from enum import Enum

from data_augmentation import train_transform, test_transform

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
    
class MaskTrainDataset(Dataset):
    
    def __init__(self, label_df, image_dir_paths, transform=None):
        super(Dataset, self).__init__()
        self.df = label_df
        self.img_dir = image_dir_paths
        self.transform = transform
        self.img_label = self.get_label()
    
    def get_label(self):
        label = self.df['mask'].values
        return label 
    
    def __getitem__(self, index):
        img_path = self.img_dir[index]
        ext = get_ext(CFG.img_dir, img_path)
        img = cv2.imread(img_path+ext, cv2.IMREAD_COLOR)
        img = img[193:193+195, 102:102+176]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            img = self.transform(image=img)['image']
        
        if self.img_label is not None:
            label = self.img_label[index]
            return img, label
    
    def __len__(self):
        return len(self.img_dir)
    
    
if __name__ == '__main__':
    # os.getcwd() = /opt/ml/project/
    df = pd.read_csv(os.path.join(os.getcwd(), "input", "data", "train", "train.csv"))
    label_df = pd.read_csv(os.path.join(os.getcwd(), "input", "data", "train", "label.csv"))
    
    data_feeding = []
    image_path = []
    image_label = []
    image_folder_path = os.path.join(os.getcwd(), "input", "data", "train", "images") # /opt/ml/project/input/data/train/images
    image_folder = os.listdir(image_folder_path) # image_folder = ['00000', '00001', '00002', ...]
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
    print(label_df['mask5'].values)        
    
    print(f'image_path[0] {image_path[0]}')
    print(f'image_folder[0] {image_folder[0]}')
    print(f'image_folder_path {image_folder_path}')
    # print(image_label[0])
    print(os.listdir(image_path[0]))
    print(os.path.join(image_path[0], os.listdir(image_path[0])[0]))
    
    print(data_feeding[0])
    
    # for i in os.listdir(image_path[0]):
    #     label = i.split('.jpg')[0]
    #     print(label)
        
    #     image_label.append(label_df[label_df['image_path'] == image_path[0]][label].values[0])
    
    # print(image_label)
    
    ext = get_ext(CFG.img_dir, image_path[0]) # /opt/ml/project/input/data/train/images, /opt/ml/project/input/data/train/images/000001_female_Asian_45
    print(ext)
    # for file_name in image_folder:
    #     image_path.append(os.path.join(image_folder_path, file_name))
    #     image_label.append(label_df[df['path'] == file_name]['mask'].values[0])