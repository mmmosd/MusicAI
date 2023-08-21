import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, w, h):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        self.w = w
        self.h = h
        self.linear = nn.Sequential(
            nn.Linear(in_features=100, out_features=int((w*h)/(16*16))),
            nn.BatchNorm1d(int((w*h)/(16*16))),
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=32*8, kernel_size=4, stride=2, padding=1,  bias=False),
            nn.BatchNorm2d(num_features=32*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32*8, out_channels=32*4, kernel_size=4, stride=2, padding=1,  bias=False),
            nn.BatchNorm2d(num_features=32*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32*4, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, inputs):
        x = self.linear(inputs).view(-1, 1, int(self.h/16), int(self.w/16))
        x = self.main(x).view(-1, 1, self.h, self.w)
        return x

# 모델을 생성하고 예시 입력 데이터로 테스트
model = Generator(1, 128, 1280)
inputs = torch.randn((32, 100))  # 입력 데이터 예시
outputs = model(inputs)
print(outputs.shape)
