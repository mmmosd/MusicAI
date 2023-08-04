import torch
import torch.nn as nn
import numpy as np
import random
import converter
import dataMaker

from torch.optim import Adam
from torch.utils.data import DataLoader

AUDIOLEN = 15
LR = 0.0002
SEED = 1

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
            kernel_size=4, stride=2, padding=1,
            bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32*2, 
            kernel_size=4, stride=2, padding=1, 
            bias=False),
            nn.BatchNorm2d(num_features=32*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=32*2, out_channels=32*4, 
            kernel_size=4, stride=2, padding=1, 
            bias=False),
            nn.BatchNorm2d(num_features=32*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=32*4, out_channels=32*8, 
            kernel_size=4, stride=2, padding=1, 
            bias=False),
            nn.BatchNorm2d(num_features=32*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=32*8, out_channels=1, 
            kernel_size=(8, 80), stride=1, padding=0, 
            bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.main(inputs)
        o = self.final_layer(x)
        return o.view(-1, 1)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=32*8, 
            kernel_size=(8, 80), stride=1, padding=0, 
            bias=False),
            nn.BatchNorm2d(num_features=32*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32*8, out_channels=32*4, 
            kernel_size=4, stride=2, padding=1, 
            bias=False),
            nn.BatchNorm2d(num_features=32*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32*4, out_channels=32*2, 
            kernel_size=4, stride=2, padding=1, 
            bias=False),
            nn.BatchNorm2d(num_features=32*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32*2, out_channels=32, 
            kernel_size=4, stride=2, padding=1, 
            bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),
        )
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=1, 
            kernel_size=4, stride=2, padding=1, 
            bias=False),
            nn.Tanh()
        )

    def forward(self, inputs):
        inputs = inputs.view(-1, 100, 1, 1)
        x = self.main(inputs)
        o = self.final_layer(x)
        return o
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def Train(epoch, batch_size, saving_interval, ngpu):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    random.seed(SEED)

    DataList = dataMaker.Load_Data_As_Spectrogram(AUDIOLEN)
    dataloader = DataLoader(DataList, batch_size=batch_size, shuffle=True)

    D = Discriminator(ngpu).to(device)
    G = Generator(ngpu).to(device)

    D.apply(weights_init)
    G.apply(weights_init)

    criterion = nn.BCELoss()

    G_optimizer = Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    D_optimizer = Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    for epoch in range(epoch):
        for real_data in dataloader:
            batch_size = real_data.shape[0]

            target_real = torch.ones(batch_size, 1)
            target_fake = torch.zeros(batch_size, 1)

            z = torch.randn((batch_size, 100)) # 랜덤 벡터 z

            # train D
            D_loss = criterion(D(real_data), target_real) + criterion(D(G(z)), target_fake) # 판별자의 오차 = 진짜를 판별했을 때 오차 + 가짜를 판별했을 때 오차

            D_optimizer.zero_grad() # 역전파 시 기울기 소실 방지
            D_loss.backward()
            D_optimizer.step()
            
            # train G
            G_loss = criterion(D(G(z)), target_real) # 생성자의 오차 = 판별자가 가짜를 판별했을 때의 확률과 진짜 확률과의 오차

            G_optimizer.zero_grad() # 역전파 시 기울기 소실 방지
            G_loss.backward()
            G_optimizer.step()

            print('epoch: {}, G_loss: {}, D_loss: {}, D(G(z)): {}'.format(epoch+1, G_loss, D_loss, D(G(z))[0].detach().numpy()))

        if (epoch%saving_interval == 0):
            z = torch.randn((batch_size, 100))
            Gresult = G(z)
            Gresult = Gresult[0].detach().numpy().reshape(128, 1280)

            converter.Save_Spectrogram_To_Audio(Gresult, 'epoch_{}'.format(epoch+1))
            converter.Save_Spectrogram_To_Image(Gresult, 'epoch_{}'.format(epoch+1))

    z = torch.randn((batch_size, 100))
    Gresult = G(z)
    Gresult = Gresult[0].detach().numpy().reshape(128, 1280)

    converter.Save_Spectrogram_To_Audio(Gresult, 'result')
    converter.Save_Spectrogram_To_Image(Gresult, 'result')

Train(20, 32, 1, 1)