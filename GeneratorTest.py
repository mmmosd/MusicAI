import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, w, h):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=32*8, kernel_size=4, stride=2, padding=1,  bias=False),
            nn.BatchNorm2d(num_features=32*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32*8, out_channels=32*4, kernel_size=4, stride=2, padding=1,  bias=False),
            nn.BatchNorm2d(num_features=32*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32*4, out_channels=32*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32*2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),

            # nn.Linear(in_features=32*16*16, out_features=w*h),
            # nn.Tanh()
        )

    def forward(self, inputs, w, h):
        inputs = inputs.view(-1, 100, 1, 1)
        x = self.main(inputs)
        return x.view(-1, 1, h, w)

# 모델을 생성하고 예시 입력 데이터로 테스트
model = Generator(1, 128, 1280)
inputs = torch.randn(100, 128, 1280)  # 입력 데이터 예시
outputs = model(inputs)
print(outputs.shape)
