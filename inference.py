from Dataset.MaskDataset import MaskDataset
from utils.model_manage import load_checkpoint
import torch
from easydict import EasyDict
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
from Models.EfficientNet_b0 import EfficientNet_b0
from Models.EfficientNet_b1 import EfficientNet_b1
from Models.EfficientNet_b2 import EfficientNet_b2

if __name__ == "__main__":
    config = EasyDict({
        'image_root_path': "/opt/ml/input/data/eval/images",
        'data_csv_path': '/opt/ml/input/data/eval/info.csv',
        'batch_size': 64,
        'Train_type': ("Train", "Validation", "Test"),
        'seed': 444,
        'image_size': (512, 384),
        'desc': 'Normalize_FL_b1',
        'mean': [0.485, 0.456, 0.406],  ## mask: [0.558, 0.512, 0.478], imageNet: [0.485, 0.456, 0.406]
        'std': [0.229, 0.224, 0.225],   ## mask: [0.218, 0.238, 0.252], imageNet: [0.229, 0.224, 0.225]
        'checkpoint_path': '/opt/ml/code/runs/2022-10-27-024241-EfficientNet_b1-Normalize_FL_b1'
    })

    transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)  
    ])  


    test_dataset = MaskDataset(
        image_root_path = config.image_root_path,
        data_csv_path = config.data_csv_path, 
        split_rate = 0, 
        train_type = config.Train_type[2],
        is_inference = True,
        transform = transforms,
        is_soft_label = False
    )

    test_dataloader = DataLoader(test_dataset, config.batch_size, shuffle=False, num_workers=4, drop_last=False)
    model = EfficientNet_b1()
    optimizer = torch.optim.Adam(model.parameters())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    load_checkpoint(torch.load(f'{config.checkpoint_path}/checkpoint.pth.tar'), model, optimizer)
    submission = pd.read_csv(config.data_csv_path)

    with torch.no_grad():
        model.eval()
        all_predictions = []
        
        for images in tqdm(test_dataloader):
            images = images.to(device)
            predict = model(images)
            predict = predict.argmax(dim=-1)
            all_predictions.extend(predict.detach().cpu().numpy())

    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join(config.checkpoint_path, 'submission.csv'), index=False)
    print('test inference is done!')