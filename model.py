import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import converter
import numpy as np
import glob
import random
import dataMaker

from torch import nn

SAMPLE_DATA = './Sample_Data/*'
EPOCHS = 20
AUDIOLEN = 15
LR = 2e-4
SEED = 1
random.seed(SEED)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.main(x)
    
class Generator(nn.Module):
    def __init__(self,in_dim):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(in_dim, 8*8),
            nn.ReLU(inplace=True), 
            nn.Linear(8*8, 8*8*2),
            nn.ReLU(inplace=True),         
            nn.Linear(8*8*2, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

def Train():
    X_list, height, width = dataMaker.Load_Data_As_Spectrogram(AUDIOLEN)