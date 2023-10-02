import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import converter
import dataMaker
import gc

from torch.optim import Adam
from torch.utils.data import DataLoader

DATACOUNT = 40000
BATCH_SIZE = 64
LR = 0.0002
NUM_EPOCH = 100

class Discriminator(nn.Module):
    def __init__(self, w, h):
        super(Discriminator, self).__init__()

        # 보통의 CNN 구조에서는 Max-Pooling으로 이미지를 크기를 줄여나가지만, Max-Pooling은 미분이 불가능하여 학습을 할 수 없기 때문에 strided convolution을 사용함

        self.w = w
        self.h = h

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=64*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*2),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64*2, out_channels=64*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*4),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64*4, out_channels=64*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*8),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64*8, out_channels=64*16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*16),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5)
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=64*16, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5)
        )
        self.linear = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features=int((w*h)/(32*32)), out_features=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = inputs.view(-1, 1, self.h, self.w)
        x = self.main(inputs)
        x = self.conv1x1(x)
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
            nn.ReLU(True),
        )
        self.convT1x1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=64*16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=64*16),
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64*16, out_channels=64*8, kernel_size=4, stride=2, padding=1, bias=False),
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
        x = self.convT1x1(x)
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

def Add_Guassian_Noise(tensor, mean, std, device):
    noise = torch.randn((tensor.size()), device=device) * std + mean
    return tensor + noise

def Train(img_saving_interval, model_saving_interval, save_img_count, continue_learning=False):
    DataList, w, h = dataMaker.Load_Data_As_Spectrogram(5, max_count=DATACOUNT, shuffle=False)
    dataloader = DataLoader(DataList, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print("data_shape: {}, {}".format(w, h))
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print('GPU count: {}'.format(torch.cuda.device_count()))
    
    if continue_learning:
        D = torch.jit.load('Discriminator.pt')
        G = torch.jit.load('Generator.pt')
    else:
        D = Discriminator(w, h).to(device)
        G = Generator(w, h).to(device)

    # 가중치 초기화
    D.apply(weights_init)
    G.apply(weights_init)

    criterion = nn.BCELoss() # 손실 함수 Binary Cross Entropy (실제값이 1일 때, 예측값이 0에 가까울수록 오차가 커짐)

    # 최적화 함수 (beta1은 0.5, learning rate는 0.0002)
    G_optimizer = Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    D_optimizer = Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    #쓰레기값 제거
    gc.collect()
    torch.cuda.empty_cache()

    D_loss_graph = []
    G_loss_graph = []
    Discrime_real = []
    Discrime_fake = []

    for epoch in range(1, NUM_EPOCH+1):
        for real_data in dataloader:
            real_data = real_data.detach().to(device)

            target_real = torch.ones(BATCH_SIZE, 1, device=device)
            target_fake = torch.zeros(BATCH_SIZE, 1, device=device)

            z = torch.randn((BATCH_SIZE, 100), device=device) # 랜덤 벡터 z (z의 값을 조정하여 원하는 결과물을 얻을 수 있음)

            # train D
            # 판별자가 진짜 데이터를 판별하면 1(target_real)로, 가짜 데이터를 판별하면 0(target_fake)에 가까워지게끔 훈련, 즉 진짜 데이터와 가짜 데이터를 1과 0으로 구분하도록 훈련함
            D.zero_grad()

            fake_data = G(z)
            D_loss = (criterion(D(real_data), target_real) + criterion(D(fake_data), target_fake)) / 2

            D_loss.backward()
            D_optimizer.step()
            
            # train G
            # 판별자가 생성자의 출력물을 판별할 때 1(target_real)에 가까워지게끔 생성자를 훈련, 즉 판별자가 생성자의 출력물을 진짜 데이터(1)로 구분하게끔 훈련
            G.zero_grad()

            fake_data = G(z)
            G_loss = criterion(D(fake_data), target_real)

            G_loss.backward()
            G_optimizer.step()

            # 기대하는 훈련 방향
            # 훈련을 진행할수록 생성자는 판별자가 진짜 데이터로 인식하게끔 하는 결과물을 도출해낸다.
            # 반대로 판별자는 생성자의 출력물과 진짜 데이터를 잘 구분하지 못하는 상황이 나타난다.
            
            d_real = D(real_data).cpu().detach().numpy().mean()
            d_fake = D(fake_data).cpu().detach().numpy().mean()
            D_loss = D_loss.cpu().detach().numpy()
            G_loss = G_loss.cpu().detach().numpy()

            D_loss_graph.append(D_loss)
            G_loss_graph.append(G_loss)
            Discrime_real.append(d_real)
            Discrime_fake.append(d_fake)

            print('epoch: {}, D_loss: {}, G_loss: {}, D(G(z)): {}, D(real_data): {}'.format(epoch, D_loss, G_loss, d_fake, d_real))

        if (epoch%model_saving_interval == 0):
            D_scripted = torch.jit.script(D)
            G_scripted = torch.jit.script(G)
            
            D_scripted.save('Discriminator_epoch_{}'.format(epoch)+'.pt')
            G_scripted.save('Generator_epoch_{}'.format(epoch)+'.pt')

        if (epoch%img_saving_interval == 0):
            z = torch.randn((save_img_count, 100), device=device)
            save_Result(G(z), 'epoch_{}'.format(epoch))

    plt.subplot(2,2,1)
    plt.plot(D_loss_graph)
    plt.title('Discriminator')
    plt.subplot(2,2,2)
    plt.plot(G_loss_graph)
    plt.title('Generator')

    plt.subplot(2,2,3)
    plt.plot(Discrime_real)
    plt.title('Discriminator_real_data')
    plt.subplot(2,2,4)
    plt.plot(Discrime_fake)
    plt.title('Discriminator_fake_data')

    plt.show()
    
    z = torch.randn((save_img_count, 100), device=device)
    save_Result(G(z), 'result')

def Generate(saved_model_name, interpolation_count=3, volume=25):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # 저장된 클래스를 불러왔기 때문에 모델 선언을 하지 않아도 됨
    G = torch.jit.load(saved_model_name)
    G.eval()

    z = torch.randn((2, 100), device=device)
    z_interpolation = torch.Tensor(np.linspace(z[0].cpu().detach().numpy(), z[1].cpu().detach().numpy(), interpolation_count)).to(device)

    result = G(z_interpolation).cpu().detach().numpy()

    for i in range(interpolation_count):
        spg = result[i][0]
        audio = converter.Save_Spectrogram_To_Audio(spg, filename='none', volume=volume, write=False)

        if (i == 0):
            y = np.array(audio)
        else:
            y = np.concatenate((y, audio), axis=0)

    return y

def save_Result(G_result, save_name):
    for i in range(G_result.size(dim=0)):
        spg = G_result[i][0].cpu().detach().numpy()
        converter.Save_Spectrogram_To_Audio(spg, save_name+'_{}'.format(i))
        converter.Save_Spectrogram_To_Image(spg, save_name+'_{}'.format(i))


# Train(img_saving_interval=1, model_saving_interval=5, save_img_count=3, continue_learning=False)