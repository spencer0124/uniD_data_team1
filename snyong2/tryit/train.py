# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from model import DnCNN  

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(torch.tensor(PIXEL_MAX)) - 10 * torch.log10(mse)

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
    BATCH_SIZE = 4
    EPOCHS = 2
    LEARNING_RATE = 0.001
    
    # Device configuration
    device = torch.device('cuda')
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Dataset and DataLoader
    train_dataset = DenoisingDataset(
        clean_dir='/home/work/.default/hyunwoong/Contest/event/Training/clean',
        noisy_dir='/home/work/.default/hyunwoong/Contest/event/Training/noisy',
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Model, Loss, Optimizer
    model = DnCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    
    # Training loop
    total_steps = len(train_loader)
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        for i, (noisy_images, clean_images) in enumerate(train_loader):
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)
            
            with autocast():
                outputs = model(noisy_images)
                loss = criterion(outputs, clean_images)
                psnr = calculate_psnr(outputs, clean_images)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{total_steps}], '
                      f'Loss: {loss.item():.4f}, PSNR: {psnr.item():.2f}')
        
        print(f'Epoch [{epoch+1}/{EPOCHS}] completed')
    
    end_time = time.time()
    print(f'Total training time: {end_time - start_time:.2f} seconds')
    
    # Save model
    torch.save(model.state_dict(), 'denoiser_model.pth')

if __name__ == '__main__':
    train()