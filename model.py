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
    def __init__(self, ngpu, w, h):
        super(Discriminator, self).__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(in_features=w*h, out_features=512),
            nn.LeakyReLU(0.02, True),
            nn.Dropout(0.1),

            nn.Linear(in_features=512, out_features=512*2),
            nn.LeakyReLU(0.02, True),
            nn.Dropout(0.1),

            nn.Linear(in_features=512*2, out_features=512*4),
            nn.LeakyReLU(0.02, True),
            nn.Dropout(0.1),

            nn.Linear(in_features=512*4, out_features=512*8),
            nn.LeakyReLU(0.02, True),
            nn.Dropout(0.1),

            nn.Linear(in_features=512*8, out_features=512*16),
            nn.LeakyReLU(0.02, True),
            nn.Dropout(0.1)
        )
        self.final_layer = nn.Sequential(
            nn.Linear(in_features=512*16, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.main(inputs)
        o = self.final_layer(x)
        return o.view(-1, 1)

class Generator(nn.Module):
    def __init__(self, ngpu, w, h):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(in_features=100, out_features=512*16),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(in_features=512*16, out_features=512*8),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(in_features=512*8, out_features=512*4),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(in_features=512*4, out_features=512*2),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(in_features=512*2, out_features=512),
            nn.ReLU(True),
            nn.Dropout(0.1)
        )
        self.final_layer = nn.Sequential(
            nn.Linear(in_features=512, out_features=w*h),
            nn.Tanh()
        )

    def forward(self, inputs):
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

    DataList, w, h = dataMaker.Load_Data_As_Spectrogram(AUDIOLEN)
    dataloader = DataLoader(DataList, batch_size=batch_size, shuffle=True)

    print("data_shape: {}, {}".format(h, w))

    D = Discriminator(ngpu, w, h).to(device)
    G = Generator(ngpu, w, h).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        D = nn.DataParallel(D, list(range(ngpu)))
        G = nn.DataParallel(G, list(range(ngpu)))

    D.apply(weights_init)
    G.apply(weights_init)

    criterion = nn.BCELoss() # 손실 함수

    # 최적화 함수
    G_optimizer = Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    D_optimizer = Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    for epoch in range(epoch):
        for real_data in dataloader:
            batch_size = real_data.shape[0]

            target_real = torch.ones(batch_size, 1, device=device)
            target_fake = torch.zeros(batch_size, 1, device=device)

            z = torch.randn((batch_size, 100), device=device) # 랜덤 벡터 z
            real_data = real_data.reshape((batch_size, -1)).to(device)

            # train D
            D.zero_grad()

            D_loss = criterion(D(real_data), target_real) + criterion(D(G(z)), target_fake)

            D_loss.backward()
            D_optimizer.step()
            
            # train G
            G.zero_grad()

            G_loss = criterion(D(G(z)), target_real)

            G_loss.backward()
            G_optimizer.step()

            print('epoch: {}, D_loss: {}, G_loss: {}, D(G(z)): {}'.format(epoch+1, D_loss, G_loss, D(G(z))[0].detach().numpy()))

        if (epoch%saving_interval == 0):
            z = torch.randn((batch_size, 100), device=device)
            Gresult = G(z)
            Gresult = Gresult[0].detach().numpy().reshape(h, w)

            converter.Save_Spectrogram_To_Audio(Gresult, 'epoch_{}'.format(epoch+1))
            converter.Save_Spectrogram_To_Image(Gresult, 'epoch_{}'.format(epoch+1))

    z = torch.randn((batch_size, 100), device=device)
    Gresult = G(z)
    Gresult = Gresult[0].detach().numpy().reshape(h, w)

    converter.Save_Spectrogram_To_Audio(Gresult, 'result')
    converter.Save_Spectrogram_To_Image(Gresult, 'result')


Train(200, 32, 5, 1)