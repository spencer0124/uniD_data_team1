# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler
import numpy as np
from PIL import Image
import os
import time
from model import SimpleDenoiser

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return torch.tensor(100.0).to(mse.device)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

class DenoisingDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.transform = transform
        self.clean_files = sorted(os.listdir(clean_dir))
        self.noisy_files = sorted(os.listdir(noisy_dir))

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_img = Image.open(os.path.join(self.clean_dir, self.clean_files[idx])).convert('RGB')
        noisy_img = Image.open(os.path.join(self.noisy_dir, self.noisy_files[idx])).convert('RGB')
        
        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)
        
        return noisy_img, clean_img

def train():
    # Hyperparameters
    BATCH_SIZE = 8
    EPOCHS = 1
    LEARNING_RATE = 0.0001
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Dataset and DataLoader
    train_dataset = DenoisingDataset(
        clean_dir='/home/work/.default/hyunwoong/Contest/event/Training/clean',
        noisy_dir='/home/work/.default/hyunwoong/Contest/event/Training/noisy',
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Model, Loss, Optimizer
    model = SimpleDenoiser().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = GradScaler('cuda')
    
    # Training loop
    best_psnr = 0
    total_steps = len(train_loader)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_psnr = 0
        epoch_loss = 0
        
        for i, (noisy_images, clean_images) in enumerate(train_loader):
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                outputs = model(noisy_images)
                loss = criterion(outputs, clean_images)
            
            # Calculate PSNR
            with torch.no_grad():
                current_psnr = calculate_psnr(outputs, clean_images)
                epoch_psnr += current_psnr.item()
                epoch_loss += loss.item()
            
            # Backward and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if (i + 1) % 10 == 0:
                avg_psnr = epoch_psnr / (i + 1)
                avg_loss = epoch_loss / (i + 1)
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{total_steps}], '
                      f'Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}')
        
        # Epoch 평균 PSNR
        avg_epoch_psnr = epoch_psnr / len(train_loader)
        print(f'Epoch [{epoch+1}/{EPOCHS}] Average PSNR: {avg_epoch_psnr:.2f}')
        
        # Learning rate 조정
        scheduler.step(avg_epoch_psnr)
        
        # 최고 성능 모델 저장
        if avg_epoch_psnr > best_psnr:
            best_psnr = avg_epoch_psnr
            torch.save(model.state_dict(), 'best_denoiser_model.pth')
            print(f'New best model saved with PSNR: {best_psnr:.2f}')

if __name__ == '__main__':
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    
    # CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    train()