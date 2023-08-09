import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128,
            kernel_size=4, stride=2, padding=1,
            bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128*2, 
            kernel_size=4, stride=2, padding=1, 
            bias=False),
            nn.BatchNorm2d(num_features=128*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128*2, out_channels=128*4, 
            kernel_size=4, stride=2, padding=1, 
            bias=False),
            nn.BatchNorm2d(num_features=128*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128*4, out_channels=128*8, 
            kernel_size=4, stride=2, padding=1, 
            bias=False),
            nn.BatchNorm2d(num_features=128*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=128*8, out_channels=1, 
            kernel_size=4, stride=1, padding=0, 
            bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.main(inputs)
        o = self.final_layer(x)
        return o.view(-1, 1)

# 모델을 생성하고 예시 입력 데이터로 테스트
model = Discriminator(1)
inputs = torch.randn((32, 1, 128, 1280))  # 입력 데이터 예시
outputs = model(inputs)
print(outputs.shape)
