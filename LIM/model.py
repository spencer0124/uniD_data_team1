# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)  # inplace=False로 변경
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class SimpleDenoiser(nn.Module):
    def __init__(self, base_channels=64):
        super(SimpleDenoiser, self).__init__()
        
        # 초기 특징 추출
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=False)
        )
        
        # 특징 추출 블록
        self.feature_extraction = nn.ModuleList([
            ResBlock(base_channels) for _ in range(3)
        ])
        
        # 노이즈 제거 블록
        self.denoise_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=False)
            ) for _ in range(2)
        ])
        
        # 최종 복원 블록
        self.reconstruction = nn.Sequential(
            nn.Conv2d(base_channels, base_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels//2),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_channels//2, 3, kernel_size=3, padding=1)
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # 초기 특징 추출
        features = self.first_conv(x)
        
        # 특징 추출
        extracted = features
        for block in self.feature_extraction:
            extracted = block(extracted)
        
        # 노이즈 제거
        denoised = extracted
        for block in self.denoise_blocks:
            denoised = block(denoised)
        
        # Skip connection
        denoised = denoised + features
        
        # 최종 복원
        restored = self.reconstruction(denoised)
        return restored
