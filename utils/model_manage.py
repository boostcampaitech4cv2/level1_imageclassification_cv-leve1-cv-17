import torch

def save(model, optimizer, save_path):
    checkpint = {
    'state_dict' : model.state_dict(), 
    'optimizer': optimizer.state_dict(),
    }

    torch.save(checkpint, f"{save_path}/checkpoint.pth.tar")

def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print("=> Loaded checkpoint")