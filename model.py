import torch
import torch.nn as nn
import numpy as np
import random
import converter
import dataMaker

from torch.optim import Adam
from torch.utils.data import DataLoader

AUDIOLEN = 12
LR = 0.0002


class Discriminator(nn.Module):
    def __init__(self, w, h):
        super(Discriminator, self).__init__()

        self.w = w
        self.h = h
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=64, out_channels=64*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=64*2, out_channels=64*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=64*4, out_channels=64*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=64*8, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features=int((w*h)/(32*32)), out_features=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = inputs.view(-1, 1, self.h, self.w)
        x = self.main(inputs)
        x = self.linear(x).view(-1, 1)
        return x

class Generator(nn.Module):
    def __init__(self, w, h):
        super(Generator, self).__init__()

        self.w = w
        self.h = h
        self.linear = nn.Sequential(
            nn.Linear(in_features=100, out_features=int((w*h)/(32*32))),
            nn.BatchNorm1d(num_features=int((w*h)/(32*32))),
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=64*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64*8, out_channels=64*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64*4, out_channels=64*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64*2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, inputs):
        x = self.linear(inputs).view(-1, 1, int(self.h/32), int(self.w/32))
        x = self.main(x)
        return x
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) # 평균은 0, 분산은 0.02가 되도록 convolutional layer의 가중치를 랜덤하게 초기화함
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) # 평균은 0, 분산은 0.02가 되도록 linear layer의 가중치를 랜덤하게 초기화함
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02) # 평균은 1, 분산은 0.02가 되도록 batchnormalization layer의 가중치를 랜덤하게 초기화함
        nn.init.constant_(m.bias.data, 0)

def Train(epoch, batch_size, saving_interval, save_img_count):
    DataList, w, h = dataMaker.Load_Data_As_Spectrogram(AUDIOLEN)
    dataloader = DataLoader(DataList, batch_size=batch_size, shuffle=True, drop_last=True)
    print("data_shape: {}, {}".format(w, h))
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print('GPU count: {}'.format(torch.cuda.device_count()))
    
    D = Discriminator(w, h).to(device)
    G = Generator(w, h).to(device)

    # 가중치 초기화
    D.apply(weights_init)
    G.apply(weights_init)

    print(G.state_dict())

    if (torch.cuda.device_count() > 1):
        D = nn.DataParallel(D)
        G = nn.DataParallel(G)

    criterion = nn.BCELoss() # 손실 함수 Binary Cross Entropy (실제값이 1일 때, 예측값이 0에 가까울수록 오차가 커짐)

    # 최적화 함수 (beta1은 0.5, learning rate는 0.0002)
    G_optimizer = Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    D_optimizer = Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    for epoch in range(epoch+1):
        for real_data in dataloader:
            real_data = real_data.to(device)
            target_real = torch.ones(batch_size, 1, device=device)
            target_fake = torch.zeros(batch_size, 1, device=device)

            z = torch.randn((batch_size, 100), device=device) # 랜덤 벡터 z (z의 값을 조정하여 원하는 결과물을 얻을 수 있음)

            # train D
            # 판별자가 진짜 데이터를 판별하면 1(target_real)로, 가짜 데이터를 판별하면 0(target_fake)에 가까워지게끔 훈련, 즉 진짜 데이터와 가짜 데이터를 1과 0으로 구분하도록 훈련함
            D.zero_grad()

            D_loss = (criterion(D(real_data), target_real) + criterion(D(G(z)), target_fake)) / 2

            D_loss.backward()
            D_optimizer.step()
            
            # train G
            # 판별자가 생성자의 출력물을 판별할 때 1(target_real)에 가까워지게끔 생성자를 훈련, 즉 판별자가 생성자의 출력물을 진짜 데이터(1)로 구분하게끔 훈련
            G.zero_grad()

            G_loss = criterion(D(G(z)), target_real)

            G_loss.backward()
            G_optimizer.step()

            # 기대하는 훈련 방향
            # 훈련을 진행할수록 생성자는 판별자가 진짜 데이터로 인식하게끔 하는 결과물을 도출해낸다.
            # 반대로 판별자는 생성자의 출력물과 진짜 데이터를 잘 구분하지 못하는 상황이 나타난다.

            print('epoch: {}, D_loss: {}, G_loss: {}, D(G(z)): {}, D(real_data): {}'.format(epoch, D_loss, G_loss, D(G(z))[0].cpu().detach().numpy(), D(real_data)[0].cpu().detach().numpy()))

        G_scripted = torch.jit.script(G)
        D_scripted = torch.jit.script(D)

        G_scripted.save('Generator.pt')
        D_scripted.save('Discriminator.pt')

        if (epoch%saving_interval == 0):
            z = torch.randn((save_img_count, 100), device=device)
            save_Result(G(z), 'epoch_{}'.format(epoch))
    
    z = torch.randn((save_img_count, 100), device=device)
    save_Result(G(z), 'result')

def Generate_Music(save_name, volume=15):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    G = torch.jit.load('Generator.pt')
    G.eval()

    z = torch.randn((1, 100), device=device)

    result = G(z)

    spg = result[0].cpu().detach().numpy()
    audio = converter.Save_Spectrogram_To_Audio(spg, save_name, volume=volume, write=False)

    return audio


def save_Result(G_result, save_name):
    for i in range(G_result.size(dim=0)):
        spg = G_result[i][0].cpu().detach().numpy()
        converter.Save_Spectrogram_To_Audio(spg, save_name+'_{}'.format(i))
        converter.Save_Spectrogram_To_Image(spg, save_name+'_{}'.format(i))

# Train(epoch=100, batch_size=16, saving_interval=1, save_img_count=3)