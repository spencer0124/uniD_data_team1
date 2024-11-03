import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Compose, ToPILImage
from PIL import Image
from model import SimpleDenoiser

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

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

def test():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Transform
    test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Test directory paths
    test_dir = '/home/work/.default/hyunwoong/Contest/repo/test/test/Input'
    output_dir = '/home/work/.default/hyunwoong/Contest/repo/test/test/sample_submission'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = SimpleDenoiser().to(device)
    model.load_state_dict(torch.load('denoiser_model.pth'))
    model.eval()
    
    # Dataset and DataLoader
    test_dataset = CustomDatasetTest(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Total test images: {len(test_dataset)}")
    
    # Process test images
    with torch.no_grad():
        for i, (noisy_image, noisy_image_path) in enumerate(test_loader):
            # Move to device and denoise
            noisy_image = noisy_image.to(device)
            denoised_image = model(noisy_image)
            
            # Post-process
            denoised_image = denoised_image.cpu().squeeze(0)
            denoised_image = (denoised_image * 0.5 + 0.5).clamp(0, 1)  # Denormalize
            denoised_image = ToPILImage()(denoised_image)
            
            # Save result
            output_filename = noisy_image_path[0]
            denoised_filename = os.path.join(output_dir, 
                                           output_filename.split('/')[-1][:-4] + '.jpg')
            denoised_image.save(denoised_filename)
            
            print(f'Processed and saved image {i+1}/{len(test_dataset)}: {denoised_filename}')

if __name__ == '__main__':
    test()