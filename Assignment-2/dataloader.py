from export import load_cifar10_data
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# Get the separated training and test data
(train_images, train_labels), (test_images, test_labels), class_names = load_cifar10_data()

print(f"Training images shape: {train_images.shape}") 
print(f"Training labels shape: {train_labels.shape}") 
print(f"Test images shape: {test_images.shape}")     
print(f"Test labels shape: {test_labels.shape}")

class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CIFAR10Dataset(train_images, train_labels, transform=transform)
test_dataset = CIFAR10Dataset(test_images, test_labels, transform=transform)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Train loader: {train_loader}")
print(f"Test loader: {test_loader}")