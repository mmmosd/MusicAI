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

            nn.Conv2d(in_channels=64*4, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features=int((w*h)/(16*16)), out_features=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = inputs.view(-1, 1, self.h, self.w)
        x = self.main(inputs)
        x = self.linear(x).view(-1, 1)
        return x

class Generator(nn.Module):
    def __init__(self, ngpu, w, h):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        self.w = w
        self.h = h
        self.linear = nn.Sequential(
            nn.Linear(in_features=100, out_features=int((w*h)/(16*16))),
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=64*4, kernel_size=4, stride=2, padding=1, bias=False),
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
        x = self.linear(inputs).view(-1, 1, int(self.h/16), int(self.w/16))
        x = self.main(x)
        return x

# class Discriminator(nn.Module):
#     def __init__(self, ngpu, w, h):
#         super(Discriminator, self).__init__()

#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             nn.Flatten(),

#             nn.Linear(in_features=w*h, out_features=128*4),
#             nn.LeakyReLU(0.2),

#             nn.Linear(in_features=128*4, out_features=128*2),
#             nn.LeakyReLU(0.2),

#             nn.Linear(in_features=128*2, out_features=128),
#             nn.LeakyReLU(0.2),

#             nn.Linear(in_features=128, out_features=1),
#             nn.Sigmoid()
#         )

#     def forward(self, inputs):
#         x = self.main(inputs)
#         return x

# class Generator(nn.Module):
#     def __init__(self, ngpu, w, h):
#         super(Generator, self).__init__()

#         self.ngpu = ngpu
#         self.w = w
#         self.h = h
#         self.main = nn.Sequential(
#             nn.Linear(in_features=100, out_features=128),
#             nn.LeakyReLU(0.2),

#             nn.Linear(in_features=128, out_features=128*2),
#             nn.LeakyReLU(0.2),

#             nn.Linear(in_features=128*2, out_features=128*4),
#             nn.LeakyReLU(0.2),

#             nn.Linear(in_features=128*4, out_features=w*h),
#             nn.Tanh()
#         )

#     def forward(self, inputs):
#         x = self.main(inputs)
#         return x.view(-1, self.h, self.w)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) #평균은 0, 분산은 0.02가 되도록 convolutional layer의 가중치를 랜덤하게 초기화함
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) #평균은 0, 분산은 0.02가 되도록 linear layer의 가중치를 랜덤하게 초기화함
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02) #평균은 1, 분산은 0.02가 되도록 batchnormalization layer의 가중치를 랜덤하게 초기화함
        nn.init.constant_(m.bias.data, 0)

def Train(epoch, batch_size, saving_interval, save_img_count, ngpu):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    DataList, w, h = dataMaker.Load_Data_As_Spectrogram(AUDIOLEN)
    dataloader = DataLoader(DataList, batch_size=batch_size, shuffle=True)

    print("data_shape: {}, {}".format(w, h))

    D = Discriminator(ngpu, w, h).to(device)
    G = Generator(ngpu, w, h).to(device)

    # weights initialization
    D.apply(weights_init)
    G.apply(weights_init)

    if (device.type == 'cuda') and (ngpu > 1):
        D = nn.DataParallel(D, list(range(ngpu)))
        G = nn.DataParallel(G, list(range(ngpu)))

    criterion = nn.BCELoss() # 손실 함수 (실제값이 1일 때, 예측값이 0에 가까울수록 오차가 커짐)

    # 최적화 함수 (Adam을 쓰는 것이 안정적, beta1은 0.5, learning rate는 0.0002)
    G_optimizer = Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    D_optimizer = Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    for epoch in range(epoch+1):
        for real_data in dataloader:
            batch_size = real_data.shape[0]

            target_real = torch.ones(batch_size, 1, device=device)
            target_fake = torch.zeros(batch_size, 1, device=device)

            z = torch.randn((batch_size, 100), device=device) # 랜덤 벡터 z (z의 값을 조정하여 원하는 결과물을 얻을 수 있음)

            # train D 
            # (판별자가 진짜 데이터를 판별하면 1(target_real)로, 가짜 데이터를 판별하면 0(target_fake)에 가까워지게끔 훈련, 즉 진짜 데이터와 가짜 데이터를 1과 0으로 구분하도록 훈련함)
            D.zero_grad()

            D_loss = (criterion(D(real_data), target_real) + criterion(D(G(z)), target_fake)) / 2

            D_loss.backward()
            D_optimizer.step()
            
            # train G 
            # (판별자가 생성자의 출력물을 판별할 때 1(target_real)에 가까워지게끔 생성자를 훈련, 즉 판별자가 생성자의 출력물을 진짜 데이터(1)로 판단하게끔 훈련)
            G.zero_grad()

            G_loss = criterion(D(G(z)), target_real)

            G_loss.backward()
            G_optimizer.step()

            # 훈련을 진행할수록 생성자는 판별자가 진짜 데이터로 인식하게끔 하는 결과물을 도출해낸다. (생성자의 오차가 줄어듦)
            # 반대로 판별자는 생성자의 출력물과 진짜 데이터를 잘 구분하지 못하는 상황이 나타난다. (판별자의 오차가 늘어남)

            print('epoch: {}, D_loss: {}, G_loss: {}, D(G(z)): {}, D(real_data): {}'.format(epoch, D_loss, G_loss, D(G(z))[0].detach().numpy(), D(real_data)[0].detach().numpy()))

        torch.save(G, 'Generator.pt')
        torch.save(D, 'Discriminator.pt')

        if (epoch%saving_interval == 0):
            z = torch.randn((save_img_count, 100), device=device)
            save_Result(G(z), 'epoch_{}'.format(epoch))
    
    z = torch.randn((save_img_count, 100), device=device)
    save_Result(G(z), 'result')

# def Generate_Spectrogram(count=1):
#     G = torch.load('Generator.pt')
#     z = torch.randn((count, 100))

#     result = G(z)
#     result.detach().numpy().reshape(count, )


def save_Result(G_result, save_name):
    for i in range(G_result.size(dim=0)):
        img = G_result[i][0].detach().numpy()
        converter.Save_Spectrogram_To_Audio(img, save_name+'_{}'.format(i))
        converter.Save_Spectrogram_To_Image(img, save_name+'_{}'.format(i))


Train(50, 128, 1, 3, 1)