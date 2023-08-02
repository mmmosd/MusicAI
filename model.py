import torch
import torch.nn as nn
import numpy as np
import random
import converter
import dataMaker

from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

AUDIOLEN = 15
LR = 0.0002
SEED = 1

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=28*4, 
                      kernel_size=4, stride=2, padding=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=28*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=28*4, out_channels=28*8, 
                      kernel_size=4, stride=2, padding=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=28*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=28*8, out_channels=1, 
                      kernel_size=(32, 320), stride=1, padding=0, 
                      bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.main(inputs)
        o = self.final_layer(x)
        return o.view(-1, 1)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
        nn.ConvTranspose2d(in_channels=100, out_channels=28*8, 
            kernel_size=4, stride=1, padding=0, 
            bias=False),
        nn.BatchNorm2d(num_features=28*8),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=28*8, out_channels=28*4, 
            kernel_size=4, stride=2, padding=1, 
            bias=False),
        nn.BatchNorm2d(num_features=28*4),
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=28*4, out_channels=28*2, 
            kernel_size=4, stride=2, padding=1,
            bias=False),
        nn.BatchNorm2d(num_features=28*2),
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=28*2, out_channels=28, 
            kernel_size=8, stride=4, padding=2,
            bias=False),
        nn.BatchNorm2d(num_features=28),
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=28, out_channels=1, 
            kernel_size=(4, 40), stride=(2, 20), padding=(1, 10),
            bias=False),
        nn.Tanh())

    def forward(self, inputs):
        inputs = inputs.view(-1, 100, 1, 1)
        return self.main(inputs)

def Train(epoch, batch_size, saving_interval):
    random.seed(SEED)

    DataList = dataMaker.Load_Data_As_Spectrogram(AUDIOLEN)
    dataloader = DataLoader(DataList, batch_size=batch_size, shuffle=True)

    D = Discriminator()
    G = Generator()

    criterion = nn.BCELoss()

    G_optimizer = Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    D_optimizer = Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    for epoch in range(epoch):
        for real_data in dataloader:
            print(real_data.shape)

            batch_size = real_data.shape[0]

            target_real = torch.ones(batch_size, 1)
            target_fake = torch.zeros(batch_size, 1)

            #discriminator train
            D_result_from_real = D(real_data)

            D_loss_real = criterion(D_result_from_real, target_real)

            z = torch.randn((batch_size, 100))
            fake_data = G(z)

            D_result_from_fake = D(fake_data)
            D_loss_fake = criterion(D_result_from_fake, target_fake)
            D_loss = D_loss_real + D_loss_fake

            D.zero_grad()
            D_loss.backward()
            D_optimizer.step()
            

            #generator train
            z = torch.randn((batch_size, 100))

            fake_data = G(z)

            if (epoch%saving_interval == 0):
                fake_data_np = fake_data
                fake_data_np = fake_data_np[0].detach().numpy().reshape(128, 1280)
                

                print(fake_data_np.shape)

                converter.Save_Spectrogram_To_Audio(fake_data_np, 'epoch_{}'.format(epoch))
                converter.Save_Spectrogram_To_Image(fake_data_np, 'epoch_{}'.format(epoch))

            D_result_from_fake = D(fake_data)
            G_loss = criterion(D_result_from_fake, target_real)

            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

        print('G_loss: {}, D_loss: {}'.format(G_loss, D_loss))

    z = torch.randn((batch_size, 100))
    Gresult = G(z)
    Gresult = Gresult[0].detach().numpy().reshape(128, 1280)

    converter.Save_Spectrogram_To_Audio(Gresult, 'result')
    converter.Save_Spectrogram_To_Image(Gresult, 'result')

Train(20, 32, 5)