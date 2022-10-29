# import random
# import pandas as pd
import numpy as np
# import cv2

# import torch
import torch.nn as nn


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


train_transform = A.Compose([
                            # A.Resize(384, 384),
                            # A.HorizontalFilp(p=0.5),
                            A.Normalize(mean=(0.558, 0.512, 0.478), std=(0.218, 0.238, 0.252), max_pixel_value=255.0, always_apply=False, p=1.0),
                            # A.Normalize(mean=(0.537, 0.482, 0.455), std=(0.227, 0.235, 0.243), max_pixel_value=255.0, always_apply=False, p=1.0),
                            # -> mean, std of image[145:145+244, 76:76+220]
                            ToTensorV2()
                            ])

test_transform = A.Compose([
                            # A.Resize(224, 224),
                            A.Normalize(mean=(0.558, 0.512, 0.478), std=(0.218, 0.238, 0.252), max_pixel_value=255.0, always_apply=False, p=1.0),
                            # A.Normalize(mean=(0.537, 0.482, 0.455), std=(0.227, 0.235, 0.243), max_pixel_value=255.0, always_apply=False, p=1.0),
                            # -> mean, std of image[145:145+244, 76:76+220] 
                            ToTensorV2()
                            ])



if __name__ == '__main__':
    print('data augmentation')