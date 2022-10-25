import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
import cv2

class MaskDataset(Dataset):
    def __init__(
        self, 
        data_csv_path : str, 
        split_rate : float, 
        train_type : str, ## Train, Validation, Test
        is_inference : bool,
        seed : int,
        transform : None,
        ) -> None:
        super().__init__()
        self.seed_everything(seed)
        self.data_csv_path = data_csv_path
        self.datas = []
        self.is_inference = is_inference
        self.transform = transform
        
        if self.is_inference:
            self.data_df = pd.read_csv(self.data_csv_path)['ImageID']
            self.datas = [[data] for data in self.data_df.values.tolist()]
        else:
            self.data_df = pd.read_csv(self.data_csv_path).drop(['id', 'gender', 'race', 'age', 'path'], axis=1)
            self.data_group = self.data_df.groupby('label')

            for label, group in self.data_group:
                split_idx = int(len(group)*split_rate)

                if train_type == "Train":
                    self.datas += list(zip(group['image_path'], [label]*len(group)))[:split_idx]
                elif train_type == "Validation":
                    self.datas += list(zip(group['image_path'], [label]*len(group)))[split_idx:]

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

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True