import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MaskMultiLabelDataset, TestDataset, MaskBaseDataset

import yaml
from easydict import EasyDict


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(num_classes=num_classes)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskMultiLabelDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, "images")
    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)

    # -- augmentation
    transform_module = getattr(
        import_module("dataset"), args.augmentation
    )  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize, crop_size=args.crop_size, mean=dataset.mean, std=dataset.std,
    )
    dataset.set_transform(transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            out = model(images)
            (mask_out, gender_out, age_out) = torch.split(out, [3, 2, 3], dim=1)
            pred_mask = torch.argmax(mask_out, dim=-1)
            pred_gender = torch.argmax(gender_out, dim=-1)
            pred_age = torch.argmax(age_out, dim=-1)

            pred = pred_mask * 6 + pred_gender * 3 + pred_age
            preds.extend(pred.cpu().numpy())

    info["ans"] = preds
    save_path = os.path.join(output_dir, f"output_{model_dir.split('/')[-1]}.csv")
    print(save_path)
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == "__main__":
    CONFIG_FILE_NAME = "./config/config.yaml"
    with open(CONFIG_FILE_NAME, "r") as yml_config_file:
        args = yaml.load(yml_config_file, Loader=yaml.FullLoader)
        args = EasyDict(args["valid"])

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
