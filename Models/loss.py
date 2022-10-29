from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, y, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=y.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

    

# try this: https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: Optional[Tensor] = None, # see python-typing
                 gamma: float = 0,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Invalid reduction mode: {reduction}')
        
        super(FocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        self.nll_loss = nn.NLLLoss(weight=alpha, reduction='none', ignore_index=ignore_index)
    
    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'reduction', 'ignore_index']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'[{type(self).__name__}({arg_str})]'
    
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            y = y.view(-1)
        
        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        
        if len(y) == 0:
            return torch.tensor(0.)
        
        # compute weighted cross entropy term: -alpha * log(pt)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)
        
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]
        
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma
        
        loss = focal_term * ce
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return loss
        
              
        
        
class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalCosineLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
    
    def forward(self, x, y):
        cos_loss = F.cosine_embedding_loss(x, y)
        BCE_loss = F.binary_cross_entropy_with_logits(x, y, reduntion='none')
        y = y.type(torch.long)
        at = self.alpha.gather(0, y.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return cos_loss + F_loss.mean()
    

# reference: https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/model.py
class Arcface(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599  
    def __init__(self, embedding_size=512, classnum=18, s=64, m=0.5, easy_margin=False):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))
        
        nn.init.xavier_uniform_(self.kernel)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m
        self.threshold = math.cos(math.pi - m)
        
    def l2_norm(input,axis=1):
        norm = torch.norm(input,2,axis,True)
        output = torch.div(input, norm)
        
        return output
        
    def forward(self, embeddings, label):
        # weights norm
        nB = len(embeddings)
        kernel_norm = self.l2_norm(self.kernel, axis = 0)
        cos_theta = cos_theta.clamp(-1, 1)
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0
        
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s
        
        return output


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    

if __name__ == '__main__':
    alpha = torch.Tensor([0.25])
    print(alpha)