import os
import numpy as np
import torch
import cv2
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

# get image path from input/data/train/train.csv
def get_image_path(data_dir, train=True):
    if train:
        image_path = []
        image_label = []
        df = pd.read_csv(os.path.join(data_dir, "input", "data", "train", "train.csv"))
        for i in range(len(df)):
            image_path.append(os.path.join(data_dir, "input", "data" "train", "images", df.iloc[i, 0]))
            image_label.append(df.iloc[i, 1])
        return image_path, image_label


class CustomDataset(Dataset):
    def __init__(self, image_path, transform=None):
        super(Dataset, self).__init__()
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.transform = transform
    
    def __getitem__(self,index):
        return (self.data[index], self.label[index])
    
    def __len__(self):
        return len(self.data)
    
    
if __name__ == '__main__':
    # print(get_image_path(os.getcwd() + '/input/data/train'))
    df = pd.read_csv(os.path.join(os.getcwd(), "input", "data", "train", "train.csv"))
    print(df.iloc[0, -1])
    print(os.path.join(os.getcwd(), "input", "data", "train", "train.csv"))
    image_path, image_label = get_image_path(os.getcwd())
    print(image_path[2])