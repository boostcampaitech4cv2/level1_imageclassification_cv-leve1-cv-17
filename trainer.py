from albumentations.pytorch.transforms import ToTensorV2
from easydict import EasyDict
from Dataset.MaskDataset import MaskDataset
from Models.EfficientNet_b0 import EfficientNet
import torch
from train import train, validation
from torch.utils.data import DataLoader
from torchvision import transforms

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((386, 386)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

config = EasyDict({
    'data_csv_path': '/opt/ml/input/data/train/train_added_label.csv',
    'epochs': 10,
    'batch_size': 64,
    'Train_type': ("Train", "Validation", "Test"),
    'split_rate': 0.8,
    'seed': 444,
    'learning_rate': 1e-3
})

if __name__ == "__main__":
    train_dataset = MaskDataset(
        config.data_csv_path, 
        config.split_rate, 
        config.Train_type[0],
        False,
        config.seed,
        transforms
        )

    val_dataset = MaskDataset(
        config.data_csv_path, 
        config.split_rate, 
        config.Train_type[1],
        False,
        config.seed,
        transforms
        )

    train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_dataloader = DataLoader(val_dataset, config.batch_size, shuffle=True, num_workers=4, drop_last=False)

    model = EfficientNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=config.learning_rate*0.1, mode='min')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    criterion.to(device)
    min_val_loss = 1000000

    for epoch in range(config.epochs):
        train_avg_loss, train_score = train(model, optimizer, criterion, train_dataloader, device, epoch)
        val_avg_loss, val_score = validation(model, criterion, val_dataloader, device, epoch)

        if val_avg_loss < min_val_loss:
            checkpint = {
                'state_dict' : model.state_dict(), 
                'optimizer': optimizer.state_dict(),
                }
            torch.save(checkpint, "./checkpoint.pth.tar")

            print(f"{min_val_loss} -> {val_avg_loss} decreased validation loss -> saved model-{epoch}")
            min_val_loss = val_avg_loss

        scheduler.step(val_avg_loss)