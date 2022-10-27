'''
Check which to use tesorboard or wandb

from torch.utils.tensorboard import SummaryWriter
import numpy as np

import wandb




if __name__ == '__main__':
    writer = SummaryWriter()
    
    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
'''