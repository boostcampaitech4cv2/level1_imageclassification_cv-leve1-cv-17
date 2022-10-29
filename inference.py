import os
import numpy as np
import pandas as pd
from datetime import datetime
import cv2

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Dataset.dataset import MaskTestDataset
from Dataset.data_augmentation import test_transform
from Models.model import EfficientnetB0, EfficientnetB1, EfficientnetB2

def load_model(saved_model, num_classes, device):
    # 이것도 가능? -> 이것이 더 좋은 방법일 수도 있음
    # 어.......... 그러니? 믿고 써야지
    model = torch.load_state_dict(saved_model)
    model.to(device)
    model.eval()
    return model
    

@torch.no_grad()
def inference(model, test_loader, device):
    model.eval()
    preds = []
    
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.float().to(device)
            if img is None:
                print('test img is empty!')
            
            pred = model(img)
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            
    return preds


if __name__ == '__main__':
    output_df = pd.read_csv(os.path.join(os.getcwd(), 'input', 'data', 'eval', 'info.csv'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # print(output_df.ImageID)
    # image_dir = os.path.join(os.getcwd(), 'input', 'data', 'eval', 'images')
    # image_paths = [os.path.join(image_dir, img_id) for img_id in output_df.ImageID]
    
    test_dataset = MaskTestDataset(output_df, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
            
    model = EfficientnetB2(num_classes=18).to(device)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'Models', 'model_2022-10-28 03:22:54.503305.pth')))
    model.eval()
    preds = inference(model, test_dataloader, device)
    
    output_df['ans'] = preds
    
    output_df.to_csv(os.path.join(os.getcwd(), 'input', 'data', 'eval', 'submission.csv'), index=False)
    