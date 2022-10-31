from glob import glob
import pandas as pd
from torch.utils.data import Dataset
import random
import cv2
import glob
import re

class MaskDataset_mh(Dataset):
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

        # l, a, b = cv2.split(image)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # cl = clahe.apply(l)

        # limg = cv2.merge((cl, a, b))
        # image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        if self.transform != None:
            image = self.transform(image)

        if self.is_inference:
            return image
        else:
            mask_label = self.datas[idx][1]
            age_gender_label = self.datas[idx][2]

            return image, mask_label, age_gender_label

    def data_add_label(self, datas : list):
        data_list = []

        for data in datas:
            image_paths = glob.glob(f'{self.image_root_path}/{data[-4]}/*.jpg')
            gender = data[1]
            age = data[3]

            if gender == "male":
                age_gender_label = 0
            else:
                age_gender_label = 3

            if 30 <= age < 60:
                age_gender_label += 1
            elif age >= 60:
                age_gender_label += 2

            for image_path in image_paths:
                status = image_path.split('/')[-1].split('.')[0]
                status = re.sub('[0-9]', '', status)

                if status == "mask":
                    mask_label = 0
                elif status == "incorrect_mask":
                    mask_label = 1
                else:
                    mask_label = 2
                    
                data_list.append([image_path, mask_label, age_gender_label])

        return data_list