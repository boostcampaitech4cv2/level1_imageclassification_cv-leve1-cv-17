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
    display_name = f'efficientnet-b2 with init_params, FocalLoss, AdamW, {datetime.today()}'
    
    wandb.init(entity=user, project=project, name=display_name, config=config)

