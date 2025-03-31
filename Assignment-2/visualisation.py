import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloader import test_loader

device = torch.device("cuda")

# def visualize_reconstruction(models, noise_factor, test_loader, num_images=8):
#     """Visualize original, noisy and reconstructed images from multiple models"""
#     num_models = len(models)
#     # Set figure size and adjust subplot parameters
#     fig, axes = plt.subplots(num_models * 2, num_images, figsize=(20, num_models * 2))
#     plt.subplots_adjust(wspace=0.05, hspace=0.2)  # Reduce spacing between images

#     with torch.no_grad():
#         images, _ = next(iter(test_loader))
#         images = images.to(device)

#         for idx, (model_name, model) in enumerate(models.items()):
#             model.eval()
#             model.to(device)
            
#             if model_name == 'ae':  
#                 reconstructed = model(images)
#                 images_np = images.cpu().numpy()
#                 reconstructed_np = reconstructed.cpu().numpy()
                
#                 for i in range(num_images):
#                     # Original image
#                     axes[idx * 2, i].imshow(np.transpose(images_np[i], (1, 2, 0)))
#                     axes[idx * 2, i].axis('off')
#                     # Reconstructed image with adjusted aspect
#                     axes[idx * 2 + 1, i].imshow(np.transpose(reconstructed_np[i], (1, 2, 0)))
#                     axes[idx * 2 + 1, i].axis('off')
                
#                 # Set titles for AE with adjusted padding
#                 axes[idx * 2, 0].set_title(f"Original - {model_name}", pad=5)
#                 axes[idx * 2 + 1, 0].set_title("Reconstructed", pad=5)
                
#             else:  
#                 reconstructed, noisy, _ = model(images, noise_factor)
#                 noisy_np = noisy.cpu().numpy()
#                 reconstructed_np = reconstructed.cpu().numpy()
                
#                 for i in range(num_images):
#                     # Noisy image with adjusted aspect
#                     axes[idx * 2, i].imshow(np.transpose(noisy_np[i], (1, 2, 0)))
#                     axes[idx * 2, i].axis('off')
#                     # Reconstructed image with adjusted aspect
#                     axes[idx * 2 + 1, i].imshow(np.transpose(reconstructed_np[i], (1, 2, 0)))
#                     axes[idx * 2 + 1, i].axis('off')
                
#                 # Set titles for DAE with adjusted padding
#                 axes[idx * 2, 0].set_title(f"Noisy Input (σ={noise_factor}) - {model_name}", pad=5)
#                 axes[idx * 2 + 1, 0].set_title("Reconstructed", pad=5)

#     # Adjust layout to fit images properly
#     plt.tight_layout(pad=1.5)  # Add small padding around the entire figure
#     plt.savefig(f"Assignment-2/images/reconstruction_comparison_{noise_factor}.png", 
#                 bbox_inches='tight', 
#                 dpi=300,
#                 pad_inches=0.2)  # Add small padding when saving
#     plt.show()


def epoch_vs_loss(losses):
    """Plot training and validation losses for each noise factor"""
    plt.figure(figsize=(10, 6))
    
    for noise_factor, loss in losses.items():
        plt.plot(loss['train'], label=f'Train Loss (Noise {noise_factor})')
        plt.plot(loss['val'], label=f'Val Loss (Noise {noise_factor})')
    
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig("Assignment-2/images/loss_vs_epoch.png")
    # plt.show()

def visualize_reconstruction(models, noise_factor, test_loader, num_images=8):
    """Visualize original, noisy and reconstructed images from multiple models"""
    num_models = len(models)
    # Increase figure size and reduce margins
    fig, axes = plt.subplots(num_models * 2, num_images, figsize=(20, num_models * 3))
    # Almost eliminate spacing between images
    plt.subplots_adjust(wspace=0.01, hspace=0.15)  # Slightly increased hspace for labels

    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.to(device)

        for idx, (model_name, model) in enumerate(models.items()):
            model.eval()
            model.to(device)
            
            # Add row labels on the left
            if model_name == 'ae':
                fig.text(0.02, 0.73 - (idx * 0.25), 'AutoEncoder', 
                        fontsize=12, rotation=90, va='center')
            else:
                fig.text(0.02, 0.73 - (idx * 0.25), f'DAE (σ={noise_factor})', 
                        fontsize=12, rotation=90, va='center')
            
            if model_name == 'ae':  
                reconstructed = model(images)
                images_np = images.cpu().numpy()
                reconstructed_np = reconstructed.cpu().numpy()
                
                for i in range(num_images):
                    # Remove axis spines and increase image size
                    axes[idx * 2, i].spines['top'].set_visible(False)
                    axes[idx * 2, i].spines['right'].set_visible(False)
                    axes[idx * 2, i].spines['bottom'].set_visible(False)
                    axes[idx * 2, i].spines['left'].set_visible(False)
                    axes[idx * 2, i].imshow(np.transpose(images_np[i], (1, 2, 0)))
                    axes[idx * 2, i].set_xticks([])
                    axes[idx * 2, i].set_yticks([])
                    
                    axes[idx * 2 + 1, i].spines['top'].set_visible(False)
                    axes[idx * 2 + 1, i].spines['right'].set_visible(False)
                    axes[idx * 2 + 1, i].spines['bottom'].set_visible(False)
                    axes[idx * 2 + 1, i].spines['left'].set_visible(False)
                    axes[idx * 2 + 1, i].imshow(np.transpose(reconstructed_np[i], (1, 2, 0)))
                    axes[idx * 2 + 1, i].set_xticks([])
                    axes[idx * 2 + 1, i].set_yticks([])
                
                # Modify only the titles
                axes[idx * 2, 0].set_title("Original Input", pad=2, fontsize=10)
                axes[idx * 2 + 1, 0].set_title("Reconstructed Output", pad=2, fontsize=10)
                
            else:  
                reconstructed, noisy, _ = model(images, noise_factor)
                noisy_np = noisy.cpu().numpy()
                reconstructed_np = reconstructed.cpu().numpy()
                
                for i in range(num_images):
                    # Remove axis spines and increase image size
                    axes[idx * 2, i].spines['top'].set_visible(False)
                    axes[idx * 2, i].spines['right'].set_visible(False)
                    axes[idx * 2, i].spines['bottom'].set_visible(False)
                    axes[idx * 2, i].spines['left'].set_visible(False)
                    axes[idx * 2, i].imshow(np.transpose(noisy_np[i], (1, 2, 0)))
                    axes[idx * 2, i].set_xticks([])
                    axes[idx * 2, i].set_yticks([])
                    
                    axes[idx * 2 + 1, i].spines['top'].set_visible(False)
                    axes[idx * 2 + 1, i].spines['right'].set_visible(False)
                    axes[idx * 2 + 1, i].spines['bottom'].set_visible(False)
                    axes[idx * 2 + 1, i].spines['left'].set_visible(False)
                    axes[idx * 2 + 1, i].imshow(np.transpose(reconstructed_np[i], (1, 2, 0)))
                    axes[idx * 2 + 1, i].set_xticks([])
                    axes[idx * 2 + 1, i].set_yticks([])
                
                # Modify only the titles
                axes[idx * 2, 0].set_title(f"Noisy Input", pad=2, fontsize=10)
                axes[idx * 2 + 1, 0].set_title("Denoised Output", pad=2, fontsize=10)

    # Add overall title
    fig.suptitle('Reconstruction Results: AutoEncoder vs Denoising AutoEncoder', 
                 fontsize=14, y=0.95)

    # Minimal padding around entire figure but leave space for labels
    plt.tight_layout(rect=[0.03, 0, 1, 0.95])  # Leave space for row labels and title
    plt.savefig(f"Assignment-2/images/reconstruction_comparison_{noise_factor}.png", 
                bbox_inches='tight', 
                dpi=300,
                pad_inches=0.2)  # Slightly increased padding to show labels
    # plt.show()