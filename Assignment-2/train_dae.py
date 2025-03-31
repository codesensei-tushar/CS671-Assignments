import torch
from tqdm import tqdm

def train_epoch(model, dataloader, val_loader, criterion, optimizer, device, noise_factor, n_epochs):
    """Train for one epoch and track losses"""
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f'Training Epoch {epoch+1}/{n_epochs}')
        
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, noisy_x, _ = model(data, noise_factor)
            
            # Compute reconstruction loss
            loss = criterion(reconstructed, data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_description(f'Training (loss={loss.item():.4f})')
        
        train_loss = total_loss / len(dataloader)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = evaluate(model, val_loader, criterion, device, noise_factor)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model, train_losses, val_losses

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, noise_factor):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Evaluating')
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        
        # Forward pass
        reconstructed, noisy_x, _ = model(data, noise_factor)
        
        # Compute reconstruction loss
        loss = criterion(reconstructed, data)
        total_loss += loss.item()
        pbar.set_description(f'Evaluating (loss={loss.item():.4f})')
        
    return total_loss / len(dataloader)