import torch
import converter
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

modelD = nn.Sequential([
    nn.Linear(img_size, hidden_size2),
    nn.Linear(hidden_size2, hidden_size1),
    nn.Linear(hidden_size2, hidden_size1),
    nn.LeakyReLU(0.2),
    nn.Sigmoid(),
])

modelG = nn.Sequential([
    nn.Linear(noise_size, hidden_size1),
    nn.Linear(hidden_size1, hidden_size2),
    nn.Linear(hidden_size2, img_size),
    nn.ReLU(),
    nn.Tanh(),
])