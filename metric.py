import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        
    
    def __call__(self, loss, model):
        score = -loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(loss, model)
    
    def save_checkpoint(self, loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {loss:.6f}). Saving model ...')
        # torch.save(model.state_dict(), self.path) 
        # 과연 save point 해 두는게 좋을까? 아니면 break 끝내고 다시 load 하는게 좋을까? 아니면 break 끝내고 best score 뽑아서 그거로 load 하는게 좋을까?
        self.val_loss_min = loss
        
    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path, map_location=torch.device('cuda')))


# print(sys.path())
def stratified_kFold(df, n_splits=5, random_state=42):
    pass

if __name__ == '__main__':
    print("psedo labeling start")
    early_stopping = EarlyStopping(patience=7, verbose=True, path = os.getcwd() + '/Models/saved_model/' + 'early_stopping_test.pt')