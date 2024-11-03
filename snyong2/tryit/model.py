# improved_model.py
import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, depth=17, num_channels=64, image_channels=3, kernel_size=3):
        super(DnCNN, self).__init__()
        layers = []
        # 첫 번째 레이어는 바이어스를 포함
        layers.append(nn.Conv2d(image_channels, num_channels, kernel_size, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        # 중간 레이어들은 바이어스를 제거하고 BatchNorm 추가
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(num_channels, num_channels, kernel_size, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_channels))
            layers.append(nn.ReLU(inplace=True))
        # 마지막 레이어
        layers.append(nn.Conv2d(num_channels, image_channels, kernel_size, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise  # Residual Learning

# 모델 초기화 예시
# model = DnCNN()
