import pickle
import numpy as np
from pathlib import Path

def load_cifar10_data():
    """Load and preprocess CIFAR-10 dataset"""
    # batch_files = [
    #     "Assignment-2/cifar-10-python/data_batch_1",
    #     "Assignment-2/cifar-10-python/data_batch_2",
    #     "Assignment-2/cifar-10-python/data_batch_3",
    #     "Assignment-2/cifar-10-python/data_batch_4",
    #     "Assignment-2/cifar-10-python/data_batch_5"
    # ]
    # test_file = "Assignment-2/cifar-10-python/test_batch"
    # meta_file = "Assignment-2/cifar-10-python/batches.meta"


    script_dir = Path(__file__).parent
    data_dir = script_dir / 'cifar-10-python'

    # Define file paths relative to export.py
    batch_files = [data_dir / f'data_batch_{i}' for i in range(1, 6)]
    test_file = data_dir / 'test_batch'
    meta_file = data_dir / 'batches.meta'

    def unpickle(file):
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    # Load training data
    train_images, train_labels = [], []
    for batch in batch_files:
        batch_data = unpickle(batch)
        train_images.append(batch_data[b'data'])
        train_labels.extend(batch_data[b'labels'])

    # Load test data
    test_data = unpickle(test_file)
    test_images = test_data[b'data']
    test_labels = test_data[b'labels']

    # Process training data
    train_images = np.vstack(train_images)
    train_labels = np.array(train_labels)
    train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Process test data
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Load class names
    meta = unpickle(meta_file)
    class_names = [name.decode('utf-8') for name in meta[b'label_names']]

    return (train_images, train_labels), (test_images, test_labels), class_names

if __name__ == '__main__':
    # The visualization code will only run if this file is executed directly
    import matplotlib.pyplot as plt
    import seaborn as sns

    (train_images, train_labels), (test_images, test_labels), class_names = load_cifar10_data()

    # ------------------- EDA -------------------

    print("\nDataset Summary:")
    print("-" * 50)
    print(f"Total training images: {train_images.shape[0]}")
    print(f"Total test images: {test_images.shape[0]}")
    print(f"Image shape: {train_images.shape[1:]} (Height, Width, Channels)")
    print(f"Total classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    print(f"Training images per class: {len(train_images) // len(class_names)}")
    print(f"Test images per class: {len(test_images) // len(class_names)}")

    mean = np.mean(train_images, axis=(0, 1, 2)) / 255.0
    std = np.std(train_images, axis=(0, 1, 2)) / 255.0
    print("\nTraining Dataset Statistics:")
    print("-" * 50)
    print(f"Mean per channel (RGB): {mean}")
    print(f"Std per channel (RGB): {std}")

    plt.figure(figsize=(15, 10))
    samples_per_class = 5
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    for i in range(len(class_names)):
        for j in range(samples_per_class):
            ax = plt.subplot(len(class_names), samples_per_class, i*samples_per_class + j + 1)
            idx = np.random.choice(np.where(train_labels == i)[0])
            plt.imshow(train_images[idx])
            plt.axis('off')

            if j == 0:
                ax.text(-0.5, 0.5, class_names[i], 
                       rotation=0, 
                       horizontalalignment='right',
                       verticalalignment='center',
                       transform=ax.transAxes,
                       fontsize=10,
                       fontweight='bold')

    plt.suptitle('Random Samples from Each Class (Training Data)', y=0.95)
    plt.show()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    for channel, color in enumerate(['red', 'green', 'blue']):
        sns.kdeplot(train_images[:1000, :, :, channel].ravel()/255.0, color=color)
    plt.title('RGB Distribution (Training Data)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')

    class_colors = np.zeros((len(class_names), 3))
    for i in range(len(class_names)):
        class_colors[i] = train_images[train_labels == i].mean(axis=(0,1,2))/255.0

    plt.subplot(1, 2, 2)
    bar_width = 0.25
    r = np.arange(len(class_names))
    plt.bar(r, class_colors[:,0], bar_width, label='Red', color='r', alpha=0.5)
    plt.bar(r + bar_width, class_colors[:,1], bar_width, label='Green', color='g', alpha=0.5)
    plt.bar(r + 2*bar_width, class_colors[:,2], bar_width, label='Blue', color='b', alpha=0.5)
    plt.xlabel('Classes')
    plt.ylabel('Average Value')
    plt.title('Average RGB Values per Class (Training Data)')
    plt.xticks(r + bar_width, class_names, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()
