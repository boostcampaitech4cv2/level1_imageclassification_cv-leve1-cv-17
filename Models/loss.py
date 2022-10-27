from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import math



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()
    

# try this: https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
    
    def forward(self, x, target):
        # x: (N, C)
        # target: (N)
        BCE_loss = F.binary_cross_entropy_with_logits(x, target, reduction='none')
        target = target.type(torch.long)
        at = self.alpha.gather(0, target.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss
        
        '''logprobs = F.log_softmax(x, dim=-1)
        probs = torch.exp(logprobs)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        focal_loss = self.alpha.gather(dim=0, index=target) * (1-probs)**self.gamma * nll_loss
        return focal_loss.mean()'''
        
        
class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalCosineLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
    
    def forward(self, x, target):
        cos_loss = F.cosine_embedding_loss(x, target)
        BCE_loss = F.binary_cross_entropy_with_logits(x, target, reduntion='none')
        target = target.type(torch.long)
        at = self.alpha.gather(0, target.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return cos_loss + F_loss.mean()
    

# reference: github.com/TreB1eN/InsightFace_Pytorch/blob/master/model.py
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
        
    def forward(self, embeddings, label):
        # weights norm
        nB = len(embeddings)
        kernel_norm = l2_norm(self.kernel, axis = 0)
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