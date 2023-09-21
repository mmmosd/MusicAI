import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, w, h):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=64*16, kernel_size=(int(w/32), int(h/32)), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=64*16),
            nn.ReLU(True),

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
        inputs = inputs.view(-1, 100, 1, 1)
        x = self.main(inputs)
        return x

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
G = Generator(416, 128).to(device)
z = torch.randn((2, 100), device=device)
data = G(z)

print(data.size())