'''
標準でない損失関数はここに定義
'''

import torch 
from torch import nn
import numpy as np

class SparseMSELoss(nn.Module):
    def __init__(self, beta=0.1, rho=0.3, **kwargs):
        super(SparseMSELoss, self).__init__()
        self.beta = beta
        self.rho = rho
        self.mse_loss = nn.MSELoss(**kwargs)

    def forward(self, pred, target, Z):
        mse_loss = self.mse_loss(pred, target)

        def KL(rho_j):
            return self.rho*torch.log(self.rho/rho_j) + (1-self.rho) * torch.log((1-self.rho)/(1-rho_j))


        l = Z.shape[1]
        RhoJ = Z.sum(dim=1)/l
        kl = KL(RhoJ)
        regularize = self.beta * torch.sum(kl)

        total_loss = mse_loss + regularize
        return total_loss