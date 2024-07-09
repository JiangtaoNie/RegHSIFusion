import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
import numpy as np


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class STN_Net(nn.Module):
    def __init__(self, in_dim):
        super(STN_Net, self).__init__()

        self.Backbone = nn.Sequential(
            nn.Conv2d(in_dim, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, 1, 1),
        )

        self.theta = torch.from_numpy(np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])).type(torch.cuda.FloatTensor)

    def forward(self, x):

        grid = self.Backbone(x).permute(0, 2, 3, 1)
        grid_ = F.affine_grid(
            self.theta.unsqueeze(0).repeat(x.shape[0], 1, 1),
            x.shape, align_corners=True
        )
        x = F.grid_sample(x, grid+grid_)
   
        return x

