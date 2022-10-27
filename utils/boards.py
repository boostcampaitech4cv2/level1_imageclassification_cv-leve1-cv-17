'''
Check which to use tesorboard or wandb

from torch.utils.tensorboard import SummaryWriter
import numpy as np
'''

import wandb
from datetime import datetime

def wandb_init(config):
    user = 'arislid'
    project = 'pstage_01'
    display_name = f'efficientnet-b2, {datetime.today()}'
    
    wandb.init(entity=user, project=project, name=display_name, config=config)


'''
if __name__ == '__main__':
    writer = SummaryWriter()
    
    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
'''