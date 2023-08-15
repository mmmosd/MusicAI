import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ngpu, w, h):
        super(Discriminator, self).__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=32, out_channels=32*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=32*2, out_channels=32*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=32*4, out_channels=32*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32*8),
            nn.LeakyReLU(0.2, True),

            nn.Linear(in_features=w*h, out_features=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.main(inputs)
        return x

# 모델을 생성하고 예시 입력 데이터로 테스트
model = Discriminator(1, 128, 1280)
inputs = torch.randn((32, 1, 128, 1280))  # 입력 데이터 예시
outputs = model(inputs)
print(outputs.shape)
