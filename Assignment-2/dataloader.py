from export import load_cifar10_data

# Get the separated training and test data
(train_images, train_labels), (test_images, test_labels), class_names = load_cifar10_data()

print(f"Training images shape: {train_images.shape}") 
print(f"Training labels shape: {train_labels.shape}") 
print(f"Test images shape: {test_images.shape}")     
print(f"Test labels shape: {test_labels.shape}")      