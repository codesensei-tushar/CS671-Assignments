import torch
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def train_epoch(model, dataloader, criterion, optimizer, device,epochs):
    # epochs = 20
    outputs = []
    losses = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        for batch_idx, (images, _) in enumerate(dataloader):  # Get images from train_loader
            images = images.to(device, dtype=torch.float32)  # Send to GPU/CPU

            output = model(images)
            loss = criterion(output, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Print loss every 100 batches to reduce console clutter
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print("Training Complete!")
    return model, losses

def evaluate(model, dataloader, criterion,device):
    model.eval()

    # Get test images and reconstructed images
    test_images, _ = next(iter(dataloader))  # Fetch a batch from test_loader
    test_images = test_images.to(device)

    with torch.no_grad():
        reconstructed_images = model(test_images)

    # Convert tensors to numpy for evaluation
    test_images_np = test_images.cpu().detach().numpy()
    reconstructed_images_np = reconstructed_images.cpu().detach().numpy()

    # Function to evaluate reconstruction quality
    def evaluate_reconstruction(original, reconstructed):
        batch_size = original.shape[0]
        ssim_scores, psnr_scores, mae_scores, mse_scores = [], [], [], []

        for i in range(batch_size):
            orig = np.transpose(original[i], (1, 2, 0))  # Convert (C, H, W) → (H, W, C)
            recon = np.transpose(reconstructed[i], (1, 2, 0))

            ssim_scores.append(ssim(orig, recon, channel_axis=-1, data_range=1.0, win_size=3))
            psnr_scores.append(psnr(orig, recon, data_range=1.0))
            mae_scores.append(np.mean(np.abs(orig - recon)))
            mse_scores.append(np.mean((orig - recon) ** 2))

        return {
            "SSIM": np.mean(ssim_scores),
            "PSNR": np.mean(psnr_scores),
            "MAE": np.mean(mae_scores),
            "MSE": np.mean(mse_scores),
        }

    # Get the evaluation results
    # metrics = evaluate_reconstruction(test_images_np, reconstructed_images_np)
    # print("Evaluation Results:", metrics)s