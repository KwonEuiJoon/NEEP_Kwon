from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from tqdm import tqdm
import random

import matplotlib.pyplot as plt
import math

class RatioNet(nn.Module):
    def __init__(self, opt):
        super(RatioNet, self).__init__()
        self.h = nn.Sequential(
            nn.Linear(opt.n_input, opt.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(opt.n_hidden, 1)
        ) 
    def forward(self, s):
        return self.h(s) - self.h(-s)
    
    
def train(opt, model, optim, data): ## training
    model.train()
    
    sample = data.to(opt.device)[torch.randint(0,opt.n,(opt.batch_size,))]
    
    log_ratio = model(sample)
    optim.zero_grad()
    
    if opt.alpha == 0.0:
        loss = (-1-log_ratio + torch.exp(-log_ratio)).mean()
    else:
        loss = (-((1+opt.alpha)*torch.exp(opt.alpha*log_ratio)-1)/opt.alpha + torch.exp(-(1+opt.alpha)*log_ratio)).mean()
    
    loss.backward()
    optim.step()
    
    return loss.item()


def validate(opt, model, data): ## prediction
    model.eval()

    calunit = 50000
    ret = []
    loss = 0
    with torch.no_grad():


        temp_log_ratio = []
        n = data.size()[0]
        index = 0
        with torch.no_grad():
            while index < n:
                prev_index = index
                next_index = index + calunit
                if next_index > n:
                    next_index = n
                temp = model(data.to(opt.device)[prev_index:next_index]).cpu().squeeze().numpy()
                temp_log_ratio.append(temp)
                
                index = index + calunit
                

        log_ratio = torch.tensor(np.array(temp_log_ratio).flatten()).to(opt.device)
    
        if opt.alpha == 0.0:
            loss += (-1-log_ratio + torch.exp(-log_ratio)).sum().cpu().item()
        else:
            loss += (-((1+opt.alpha)*torch.exp(opt.alpha*log_ratio)-1)/opt.alpha + torch.exp(-(1+opt.alpha)*log_ratio)).sum().cpu().item()
            
        log_ratio = log_ratio.cpu().squeeze().numpy()
        
    return log_ratio, loss



opt = Namespace()
opt.device = 'cpu' 
opt.n_input = 1
opt.n_hidden = 100 # number of nodes in hidden layers

opt.lr = 0.0001 ## learning rate
opt.wd = 5e-5
opt.seed = 100

## mean, std, sequence length of samples
opt.m = 1.0
opt.std = 1.0
opt.n = 10000
opt.batch_size= 10000

s = torch.normal(torch.tensor(opt.m), torch.tensor(opt.std), (opt.n,1))

## enforcing random seed value
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(50)
np.random.seed(opt.seed)
random.seed(opt.seed)

## generates model
opt.alpha = -0.5 ## value of alpha in alpha-divergence (alpha =0.0 corresponds to KL divergence)
model = RatioNet(opt)
model = model.to(opt.device)
optim = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.wd)


opt.n_iter = 3000 # number of training iteration
loss = []
for i in tqdm(range(1, opt.n_iter + 1)):
    l = train(opt, model, optim, s)
    
    if i%10 == 0:
        print('loss = '+str(l))
        


