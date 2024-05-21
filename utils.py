import matplotlib.pyplot as plt

# Function to visualize a few images
def visualize_images(data_loader, num_images=8):
    images_shown = 0
    plt.figure(figsize=(10, 10))
    for images in data_loader:
        for i in range(images.shape[0]):
            if images_shown >= num_images:
                break
            plt.subplot(4, 4, images_shown + 1)
            plt.imshow(images[i].permute(1, 2, 0).squeeze(), cmap='gray')
            plt.axis('off')
            images_shown += 1
        if images_shown >= num_images:
            break
    plt.show()
    
def show_images(images, num_images=16):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 2.5))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i, 0], cmap='gray')
        ax.axis('off')
    plt.show()
