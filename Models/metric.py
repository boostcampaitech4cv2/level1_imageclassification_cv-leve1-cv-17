import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold

from Dataset.dataset import MaskTestDataset, MaskTrainDataset
from Dataset.data_augmentation import train_transform
from Models.model import EfficientnetB0


sys.path.append(os.path.join(os.getcwd(), 'Dataset'))



# print(sys.path())
def stratified_kFold(df, n_splits=5, random_state=42):
    pass

if __name__ == '__main__':
    print(os.getcwd())
    print('k-fold')