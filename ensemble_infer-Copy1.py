import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset, MaskMultiLabelDataset

import yaml
from easydict import EasyDict

import numpy as np

def load_model(saved_model, num_classes, device, fold):
    model_cls = getattr(import_module("model-Copy1"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, f'best_{fold}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskMultiLabelDataset.num_classes  # 3 + 2 + 3
    model_list = []
    for fold in range(5):
        model = load_model(model_dir, num_classes, device, fold=fold).to(device)
        model_list.append(model)

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    oof_pred = None
    with torch.no_grad():
        for fold, model in enumerate(model_list):
            model.eval()
            
        for fold, model in enumerate(model_list):
            preds = []
            for idx, images in enumerate(loader):
                images = images.to(device)

                # TTA
                out = model(images)
                preds.extend(out.cpu().numpy())
            
            fold_pred = np.array(preds)

            if oof_pred is None:
                oof_pred = fold_pred / len(model_list)
            else:
                oof_pred += fold_pred / len(model_list)

    pred_all = np.split(oof_pred, [3, 5, 8], axis=1)
    pred_mask = np.argmax(pred_all[0], axis=-1) 
    pred_gender = np.argmax(pred_all[1], axis=-1) 
    pred_age = np.argmax(pred_all[2], axis=-1)
    preds = pred_mask * 6 + pred_gender * 3 + pred_age


    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    CONFIG_FILE_NAME = "./config/config.yaml"
    with open(CONFIG_FILE_NAME, "r") as yml_config_file:
        args = yaml.load(yml_config_file, Loader=yaml.FullLoader)
        args = EasyDict(args["valid"])

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
