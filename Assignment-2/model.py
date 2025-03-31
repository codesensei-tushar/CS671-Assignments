import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        """
        Denoising Autoencoder for CIFAR-10 images
        Args:
            latent_dim (int): Dimension of the latent space
        """
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4x4
            
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(),
            
            nn.Unflatten(1, (128, 4, 4)),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 8x8
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # 16x16
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),    # 32x32
            nn.Sigmoid()  # Output values between [0,1]
        )

    def add_noise(self, x, noise_factor):
        """Add Gaussian noise to input images"""
        noise = torch.randn_like(x) * noise_factor
        noisy_x = x + noise
        return torch.clamp(noisy_x, 0., 1.)
    
    def forward(self, x, noise_factor):
        """
        Forward pass
        Args:
            x (torch.Tensor): Input images [B, C, H, W]
            noise_factor (float): Standard deviation of Gaussian noise
        """
        # Add noise to input
        noisy_x = self.add_noise(x, noise_factor)
        
        # Encode noisy input
        latent = self.encoder(noisy_x)
        
        # Decode to reconstruct original
        reconstructed = self.decoder(latent)
        
        return reconstructed, noisy_x, latent
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent vectors to images"""
        return self.decoder(z)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super(Encoder, self).__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32 -> 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x16 -> 8x8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 8x8 -> 4x4
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256*4*4, latent_dim)

    def forward(self, x):
        x = self.conv_op(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256*4*4)
        self.conv_op = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # 4x4 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2), # 16x16 -> 32x32
            nn.Sigmoid() # Pixel values in range [0, 1]
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)
        x = self.conv_op(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, in_channles=3,latent_dim=256):
        super(AutoEncoder,self).__init__()
        self.encoder=Encoder(in_channles,latent_dim)
        self.decoder=Decoder(latent_dim,in_channles)
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x