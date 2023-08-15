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


class Discriminator(nn.Module):
    def __init__(self, ngpu, w, h):
        super(Discriminator, self).__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(in_features=w*h, out_features=128*8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(in_features=128*8, out_features=128*4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(in_features=128*4, out_features=128*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(in_features=128*2, out_features=128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.main(inputs)
        return x.view(-1, 1)

class Generator(nn.Module):
    def __init__(self, ngpu, w, h):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(in_features=100, out_features=128),
            nn.LeakyReLU(0.2),

            nn.Linear(in_features=128, out_features=128*2),
            nn.LeakyReLU(0.2),

            nn.Linear(in_features=128*2, out_features=128*4),
            nn.LeakyReLU(0.2),

            nn.Linear(in_features=128*4, out_features=128*8),
            nn.LeakyReLU(0.2),

            nn.Linear(in_features=128*8, out_features=w*h),
            nn.Tanh()
        )

    def forward(self, inputs):
        x = self.main(inputs)
        return x
    
def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def Train(epoch, batch_size, saving_interval, save_img_count, ngpu):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    DataList, w, h = dataMaker.Load_Data_As_Spectrogram(AUDIOLEN)
    dataloader = DataLoader(DataList, batch_size=batch_size, shuffle=True)

    print("data_shape: {}, {}".format(h, w))

    D = Discriminator(ngpu, w, h).to(device)
    G = Generator(ngpu, w, h).to(device)

    D.apply(initialize_weights)
    G.apply(initialize_weights)

    if (device.type == 'cuda') and (ngpu > 1):
        D = nn.DataParallel(D, list(range(ngpu)))
        G = nn.DataParallel(G, list(range(ngpu)))

    criterion = nn.BCELoss() # 손실 함수 (실제값이 1일 때, 예측값이 0에 가까울수록 오차가 커짐)

    # 최적화 함수
    G_optimizer = Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    D_optimizer = Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    for epoch in range(epoch):
        for real_data in dataloader:
            batch_size = real_data.shape[0]

            target_real = torch.ones(batch_size, 1, device=device)
            target_fake = torch.zeros(batch_size, 1, device=device)

            z = torch.randn((batch_size, 100), device=device) # 랜덤 벡터 z
            real_data = real_data.reshape((batch_size, -1)).to(device) # flatten data

            # train D
            D.zero_grad()

            D_loss = (criterion(D(real_data), target_real) + criterion(D(G(z)), target_fake)) / 2

            D_loss.backward()
            D_optimizer.step()
            
            # train G
            G.zero_grad()

            G_loss = criterion(D(G(z)), target_real)

            G_loss.backward()
            G_optimizer.step()

            print('epoch: {}, D_loss: {}, G_loss: {}, D(G(z)): {}, D(real_data): {}'.format(epoch, D_loss, G_loss, D(G(z))[0].detach().numpy(), D(real_data)[0].detach().numpy()))

        if (epoch%saving_interval == 0):
            z = torch.randn((save_img_count, 100), device=device)
            save_Result(G(z), 'epoch_{}'.format(epoch), w, h)
    
    z = torch.randn((save_img_count, 100), device=device)
    save_Result(G(z), 'result', w, h)

def save_Result(G_result, save_name, img_w, img_h):
    for i in range(G_result.size(dim=0)):
        temp = G_result[i].detach().numpy().reshape(img_h, img_w)
        converter.Save_Spectrogram_To_Audio(temp, save_name+'_{}'.format(i))
        converter.Save_Spectrogram_To_Image(temp, save_name+'_{}'.format(i))


Train(200, 32, 5, 3, 1)