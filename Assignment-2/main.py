import torch.optim as optim
import torch.nn as nn
from model import DenoisingAutoencoder
from dataloader import train_loader, test_loader
import torch
from train_dae import train_epoch, evaluate
from train_ae import train_epoch as train_epoch_ae, evaluate as evaluate_ae
from visualisation import visualize_reconstruction,epoch_vs_loss
from model import AutoEncoder

device = torch.device("cuda")

n_epochs = 20
noise_factors = [0.1,0.3,0.5]
model_paths = {
    'ae': 'Assignment-2/models/autoencoder.pth',
    0.1: 'Assignment-2/models/denoising_autoencoder_0.1.pth',
    0.3: 'Assignment-2/models/denoising_autoencoder_0.3.pth',
    0.5: 'Assignment-2/models/denoising_autoencoder_0.5.pth'
}

model = AutoEncoder(in_channles=3,latent_dim=256)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model, train_loss = train_epoch_ae(model, train_loader, criterion, optimizer, device,n_epochs)
val_loss = evaluate_ae(model, test_loader, criterion, device)
torch.save(model.state_dict(),'Assignment-2/models/autoencoder.pth')


model = DenoisingAutoencoder(latent_dim=128).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
losses = {}
for noise_factor in noise_factors:
    print(f"\nTraining with noise factor {noise_factor}")
    model, train_losses, val_losses = train_epoch(
        model, 
        train_loader, 
        test_loader,  # Using test_loader as validation
        criterion, 
        optimizer, 
        device, 
        noise_factor,
        n_epochs
    )

    losses[noise_factor] = {
        'train': train_losses,
        'val': val_losses
    }
 
    torch.save(model.state_dict(), f'Assignment-2/models/denoising_autoencoder_{noise_factor}.pth')


models = {}
# losses['ae'] = {'train': train_loss, 'val': val_loss}
for name, path in model_paths.items():
    model = DenoisingAutoencoder(latent_dim=128).to(device) if name != 'ae' else AutoEncoder(in_channles=3, latent_dim=256).to(device)
    model.load_state_dict(torch.load(path))
    models[str(name)] = model
# print(models.items())
# Call visualize_reconstruction
visualize_reconstruction(models,test_loader=test_loader, num_images=8)
epoch_vs_loss(losses)