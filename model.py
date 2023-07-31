import torch
import torch.nn as nn
import random
import converter
import dataMaker

from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

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
        nn.Conv2d(in_channels=28*8, out_channels=1, 
            kernel_size=7, stride=1, padding=0, 
            bias=False),
        nn.Sigmoid())

    def forward(self, inputs):
        o = self.main(inputs)
        return o.view(-1, 1)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=28*8, 
                kernel_size=7, stride=1, padding=0, 
                bias=False),
            nn.BatchNorm2d(num_features=28*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=28*8, out_channels=28*4, 
                kernel_size=4, stride=2, padding=1, 
                bias=False),
            nn.BatchNorm2d(num_features=28*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=28*4, out_channels=1, 
                kernel_size=4, stride=2, padding=1, 
                bias=False),
            nn.Tanh()
        )

    def forward(self, inputs):
        inputs = inputs.view(-1, 100, 1, 1)
        return self.main(inputs)

def Train(epoch, batch_size, saving_interval):
    random.seed(SEED)

    DataList, width, height = dataMaker.Load_Data_As_Spectrogram(AUDIOLEN)
    dataloader = DataLoader(DataList, batch_size=batch_size, shuffle=True)

    D = Discriminator()
    G = Generator()

    criterion = nn.BCELoss()

    G_optimizer = Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    D_optimizer = Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    for epoch in range(epoch):
        for real_data, _ in dataloader:
            real_data = Variable(real_data)
            target_real = Variable(torch.ones(batch_size, 1))
            target_fake = Variable(torch.zeros(batch_size, 1))

            #discriminator train
            D_result_from_real = D(real_data)

            D_loss_real = criterion(D_result_from_real, target_real)

            z = Variable(torch.randn((batch_size, 100)))
            fake_data = G(z)

            D_result_from_fake = D(fake_data)
            D_loss_fake = criterion(D_result_from_fake, target_fake)
            D_loss = D_loss_real + D_loss_fake

            D.zero_grad()
            D_loss.backward()
            D_optimizer.step()
            

            #generator train
            z = Variable(torch.randn((batch_size, 100)))
            z = z.cuda()

            fake_data = G(z)

            if (epoch%saving_interval == 0):
                converter.Save_Spectrogram_To_Audio(fake_data, 'epoch_{}'.format(epoch))
                converter.Save_Spectrogram_To_Image(fake_data, 'epoch_{}'.format(epoch))

            D_result_from_fake = D(fake_data)
            G_loss = criterion(D_result_from_fake, target_real)

            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()



# def Generate_Spectrogram():
#     z = Variable(torch.randn((batch_size, 100)))

#     return Generator(z)

Train(20, 32, 5)