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

        ## Encoder: 이미지를 latent 형태로 압축하는 모델
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

        ## Decoder: Latent로부터 이미지를 다시 재건축하는 모델
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


def Load_Data():
    dataMaker.Save_Cut_Audio(glob.glob(SAMPLE_DATA), AUDIOLEN, './data/')

    Data_list = glob.glob('./data/*')
    fileList = []

    print(Data_list.__len__())

    for i in range(Data_list.__len__()):
        spg = converter.Audio_To_Spectrogram(Data_list[i], AUDIOLEN).astype(np.float32)
        spg.resize((128, int(spg.shape[1]/128) * 128))
        fileList.append(spg)
    
    fileList = np.array(fileList)
    x, y = fileList[0].shape

    print(glob.glob(SAMPLE_DATA))
    print(x, y)

    return np.array(fileList), x, y

def Train():
    X_list, height, width = Load_Data()

dataMaker.Save_Cut_Audio(glob.glob(SAMPLE_DATA), AUDIOLEN, './data/')