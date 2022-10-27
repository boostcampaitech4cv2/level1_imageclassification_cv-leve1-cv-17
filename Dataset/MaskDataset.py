from glob import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import cv2
import glob
import re
import torch.nn.functional as F

class MaskDataset(Dataset):
    def __init__(
        self,
        image_root_path : str,  
        data_csv_path : str, 
        split_rate : float, 
        train_type : str, ## Train, Validation, Test
        is_inference : bool,
        transform : None,
        is_soft_label : bool
        ) -> None:
        super().__init__()

        self.image_root_path = image_root_path
        self.data_csv_path = data_csv_path
        self.datas = []
        self.is_inference = is_inference
        self.transform = transform
        self.is_soft_label = is_soft_label

        if self.is_inference:
            self.data_df = pd.read_csv(self.data_csv_path)['ImageID']
            self.datas = [[f'{self.image_root_path}/{data}'] for data in self.data_df.values.tolist()]
        else:
            self.data_list = pd.read_csv(self.data_csv_path).values.tolist()
            random.shuffle(self.data_list)
            self.split_idx = int(len(self.data_list)*split_rate)

            if train_type == "Train":
                self.datas += self.data_list[:self.split_idx]
            elif train_type == "Validation":
                self.datas += self.data_list[self.split_idx:]

            self.datas = self.data_add_label(self.datas)
            random.shuffle(self.datas)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        image_path = self.datas[idx][0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform != None:
            image = self.transform(image)

        if self.is_inference:
            return image
        else:
            label = self.datas[idx][1]

            return image, label

    def data_add_label(self, datas : list):
        data_list = []

        for data in datas:
            image_paths = glob.glob(f'{self.image_root_path}/{data[-4]}/*.jpg')

            for image_path in image_paths:
                status = image_path.split('/')[-1].split('.')[0]
                status = re.sub('[0-9]', '', status)

                if status == "mask":
                    label = data[-3]
                elif status == "incorrect_mask":
                    label = data[-2]
                else:
                    label = data[-1]

                if self.is_soft_label:
                    label = self.label_smoothing(torch.tensor(label))
                    
                data_list.append([image_path, label])

        return data_list

    def label_smoothing(self, label : int, alpha = 0.1, num_classes = 18):
        label = F.one_hot(label, num_classes=num_classes)
        label = label*(1-alpha) + alpha/num_classes

        return label