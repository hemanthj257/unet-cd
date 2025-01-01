import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim

class ChangeDetectionDataset(Dataset):
    def __init__(self, current_dir, past_dir, mask_dir, transform=None, mask_transform=None):
        self.current_dir = current_dir
        self.past_dir = past_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_pairs = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(past_dir, f)) and os.path.isfile(os.path.join(mask_dir, f))]
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        current_image_path = os.path.join(self.current_dir, self.image_pairs[idx])
        past_image_path = os.path.join(self.past_dir, self.image_pairs[idx])
        mask_path = os.path.join(self.mask_dir, self.image_pairs[idx])
        
        current_image = Image.open(current_image_path).convert('RGB')
        past_image = Image.open(past_image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            current_image = self.transform(current_image)
            past_image = self.transform(past_image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        input_image = torch.cat((current_image, past_image), dim=0)
        
        return input_image, mask

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.upconv4 = up_conv(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = up_conv(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = up_conv(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = up_conv(128, 64)
        self.decoder1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.final_conv(dec1))

# Specify the paths to the current, past, and mask image folders
current_image_path = '/home/hehe/fyp/dataset/augmented (later use for more accuracy)/current'
past_image_path = '/home/hehe/fyp/dataset/augmented (later use for more accuracy)/past'
mask_image_path = '/home/hehe/fyp/dataset/augmented (later use for more accuracy)/masks'

# Define the necessary preprocessing steps for images
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Define the necessary preprocessing steps for masks
mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize masks to 256x256
    transforms.ToTensor()  # Convert masks to PyTorch tensors
])

# Create an instance of the dataset
dataset = ChangeDetectionDataset(current_dir=current_image_path, past_dir=past_image_path, mask_dir=mask_image_path, transform=image_transform, mask_transform=mask_transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Check if CUDA is available and move the model to GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)

# Initialize the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, masks in dataloader:
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}')

print('Training complete')

# Save the trained model
model_save_path = 'unet_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')