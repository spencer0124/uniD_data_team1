#import
import os
import shutil

import random
import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import CenterCrop, Resize
from PIL import Image

import warnings
warnings.filterwarnings(action='ignore')

#CFG
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':16,
    'SEED':42
}

#randomseed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정
'''
#preprocessing
def Preprocess(base_dir):
    clean_dir = os.path.join(base_dir, 'clean')
    noisy_dir = os.path.join(base_dir, 'noisy')

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)


    source_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if 'GT' in dir_name:
                source_dirs.append(os.path.join(root, dir_name))

    if not source_dirs:
        raise ValueError("No directory containing 'GT' found")

    for source_dir in source_dirs:
        for filename in os.listdir(source_dir):
            if filename.endswith('.jpg'):
                shutil.move(os.path.join(source_dir, filename), os.path.join(clean_dir, filename))

    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name not in ['clean', 'noisy'] and 'GT' not in dir_name:
                current_dir = os.path.join(root, dir_name)
                for filename in os.listdir(current_dir):
                    if filename.endswith('.jpg'):
                        shutil.move(os.path.join(current_dir, filename), os.path.join(noisy_dir, filename))
                        
    
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_name not in ['clean', 'noisy']:
                shutil.rmtree(dir_path)
                        
    print('preprocessing done')

data_dir = './'
training_base_dir = os.path.join(data_dir, 'Training')
validation_base_dir = os.path.join(data_dir, 'Validation')

Preprocess(training_base_dir)
Preprocess(validation_base_dir)
'''
#customdataset
class CustomDataset(Dataset):
    def __init__(self, clean_image_paths, noisy_image_paths, transform=None):
        self.clean_image_paths = [os.path.join(clean_image_paths, x) for x in os.listdir(clean_image_paths)]
        self.noisy_image_paths = [os.path.join(noisy_image_paths, x) for x in os.listdir(noisy_image_paths)]
        self.transform = transform
        self.center_crop = CenterCrop(1080)
        self.resize = Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE']))

        # Create a list of (noisy, clean) pairs
        self.noisy_clean_pairs = self._create_noisy_clean_pairs()

    def _create_noisy_clean_pairs(self):
        clean_to_noisy = {}
        for clean_path in self.clean_image_paths:
            clean_id = '_'.join(os.path.basename(clean_path).split('_')[:-1])
            clean_to_noisy[clean_id] = clean_path
        
        noisy_clean_pairs = []
        for noisy_path in self.noisy_image_paths:
            noisy_id = '_'.join(os.path.basename(noisy_path).split('_')[:-1])
            if noisy_id in clean_to_noisy:
                clean_path = clean_to_noisy[noisy_id]
                noisy_clean_pairs.append((noisy_path, clean_path))
            else:
                pass
        
        return noisy_clean_pairs

    def __len__(self):
        return len(self.noisy_clean_pairs)

    def __getitem__(self, index):
        noisy_image_path, clean_image_path = self.noisy_clean_pairs[index]

        noisy_image = Image.open(noisy_image_path).convert("RGB")
        clean_image = Image.open(clean_image_path).convert("RGB")
        
        # Central Crop and Resize
        noisy_image = self.center_crop(noisy_image)
        clean_image = self.center_crop(clean_image)
        noisy_image = self.resize(noisy_image)
        clean_image = self.resize(clean_image)
        
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        
        return noisy_image, clean_image

#Modeling
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(self, num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[24, 48, 96, 192], num_refinement=4, expansion_factor=2.66):
        
        super(Restormer, self).__init__()

        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])
        
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])

        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr) + x
        return out

#Train
# 시작 시간 기록
start_time = time.time()

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 데이터셋 경로
noisy_image_paths = './Training/noisy'
clean_image_paths = './Training/clean'

# 데이터셋 로드 및 전처리
train_transform = Compose([
    ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 커스텀 데이터셋 인스턴스 생성
train_dataset = CustomDataset(clean_image_paths, noisy_image_paths, transform=train_transform)

# 데이터 로더 설정
num_cores = os.cpu_count()
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], num_workers=int(num_cores/2), shuffle=True)

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Restormer 모델 인스턴스 생성 및 GPU로 이동
model = Restormer().to(device)

# 손실 함수와 최적화 알고리즘 설정
optimizer = optim.AdamW(model.parameters(), lr = CFG['LEARNING_RATE'], weight_decay=1e-4)
criterion = nn.L1Loss()
scaler = GradScaler()
scheduler = CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'])

# 모델의 파라미터 수 계산
total_parameters = count_parameters(model)
print("Total Parameters:", total_parameters)

# 모델 학습
model.train()
best_loss = 1000

for epoch in range(CFG['EPOCHS']):
    model.train()
    epoch_start_time = time.time()
    mse_running_loss = 0.0
    
    for noisy_images, clean_images in train_loader:
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(noisy_images)
            mse_loss = criterion(outputs, clean_images)
        
        scaler.scale(mse_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        mse_running_loss += mse_loss.item() * noisy_images.size(0)

    current_lr = scheduler.get_last_lr()[0]
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    minutes = int(epoch_time // 60)
    seconds = int(epoch_time % 60)
    hours = int(minutes // 60)
    minutes = int(minutes % 60)

    mse_epoch_loss = mse_running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{CFG['EPOCHS']}, MSE Loss: {mse_epoch_loss:.4f}, Lr: {current_lr:.8f}")
    print(f"1epoch 훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")

    if mse_epoch_loss < best_loss:
        best_loss = mse_epoch_loss
        torch.save(model.state_dict(), 'best_Restormer.pth')
        print(f"{epoch+1}epoch 모델 저장 완료")

# 종료 시간 기록
end_time = time.time()

# 소요 시간 계산
training_time = end_time - start_time
minutes = int(training_time // 60)
seconds = int(training_time % 60)
hours = int(minutes // 60)
minutes = int(minutes % 60)

# 결과 출력
print(f"훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")

#inference

class CustomDatasetTest(data.Dataset):
    def __init__(self, noisy_image_paths, transform=None):
        self.noisy_image_paths = [os.path.join(noisy_image_paths, x) for x in os.listdir(noisy_image_paths)]
        self.transform = transform

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        noisy_image_path = self.noisy_image_paths[index]
        noisy_image = load_img(self.noisy_image_paths[index])
        
        # Convert numpy array to PIL image
        if isinstance(noisy_image, np.ndarray):
            noisy_image = Image.fromarray(noisy_image)

        if self.transform:
            noisy_image = self.transform(noisy_image)

        return noisy_image, noisy_image_path


test_transform = Compose([
    ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

model = Restormer()
model.load_state_dict(torch.load('best_Restormer.pth'))


# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# 데이터셋 경로
test_data_path = './open/test/Input'
output_path = './open/test/submission'

# 데이터셋 로드 및 전처리
test_dataset = CustomDatasetTest(test_data_path, transform=test_transform)

# 데이터 로더 설정
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

if not os.path.exists(output_path):
    os.makedirs(output_path)

# 이미지 denoising 및 저장
for noisy_image, noisy_image_path in test_loader:
    noisy_image = noisy_image.to(device)
    denoised_image = model(noisy_image)
    
    # denoised_image를 CPU로 이동하여 이미지 저장
    denoised_image = denoised_image.cpu().squeeze(0)
    denoised_image = (denoised_image * 0.5 + 0.5).clamp(0, 1)
    denoised_image = transforms.ToPILImage()(denoised_image)

    # Save denoised image
    output_filename = noisy_image_path[0]
    denoised_filename = output_path + '/' + output_filename.split('/')[-1][:-4] + '.jpg'
    denoised_image.save(denoised_filename) 
    
    print(f'Saved denoised image: {denoised_filename}')


#submmision
def zip_folder(folder_path, output_zip):
    shutil.make_archive(output_zip, 'zip', folder_path)
    print(f"Created {output_zip}.zip successfully.")

zip_folder(output_path, './submission')