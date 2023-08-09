import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, w, h):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=128*8, 
            kernel_size=4, stride=1, padding=0, 
            bias=False),
            nn.BatchNorm2d(num_features=128*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128*8, out_channels=128*4, 
            kernel_size=4, stride=2, padding=1,  
            bias=False),
            nn.BatchNorm2d(num_features=128*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128*4, out_channels=128*2, 
            kernel_size=4, stride=2, padding=1, 
            bias=False),
            nn.BatchNorm2d(num_features=128*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128*2, out_channels=128, 
            kernel_size=4, stride=2, padding=1, 
            bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
        )
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=1, 
            kernel_size=4, stride=2, padding=1, 
            bias=False),
            nn.Tanh()
        )

    def forward(self, inputs):
        inputs = inputs.view(-1, 100, 1, 1)
        x = self.main(inputs)
        o = self.final_layer(x)
        return o

# 모델을 생성하고 예시 입력 데이터로 테스트
model = Generator(1, 128, 1280)
inputs = torch.randn(100)  # 입력 데이터 예시
outputs = model(inputs)
print(outputs.shape)
