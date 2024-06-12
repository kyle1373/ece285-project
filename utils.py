import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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

def pad_to_target_size(condition, target_size):
    """
    Pads the input condition to the target size.

    Args:
        condition (torch.Tensor): The input tensor of size (1, m, n).
        target_size (tuple): The target size (M, N).

    Returns:
        torch.Tensor: The padded tensor of size (1, M, N).
    """
    _, _, m, n = condition.shape
    M, N = target_size

    pad_top = (M - m) // 2
    pad_bottom = M - m - pad_top
    pad_left = (N - n) // 2
    pad_right = N - n - pad_left

    padded_condition = F.pad(condition, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
    return padded_condition

def plot_generated_images(model, data_loader, num_images, batch_size, device='cpu'):
    model.eval()
    left_images = num_images
    with torch.no_grad():
        for batch_idx, (data, condition) in enumerate(data_loader):
            if batch_idx < 19:
                continue
            condition = condition.to(device)
            data = data.to(device)
            if left_images == 0:
                break
            
            z = torch.randn(batch_size, model.z_dim).to(device)
            cond_encoded = model.forward_condition_encoder(condition)
            sample = model.decode(z, cond_encoded).cpu()
            sample = sample.view(batch_size, 1, 64, 64)
            
            if left_images > batch_size:
                print_images = batch_size
                left_images = left_images-batch_size
            else:
                print_images = left_images
                left_images = 0

            for i in range(print_images):
                ref = data[i].cpu().detach().numpy().reshape(64, 64)
                img = sample[i].cpu().detach().numpy().reshape(64, 64)
                
                # Handle condition image with random shape
                cond = condition[i].cpu().detach().numpy()
                if len(cond.shape) > 2:
                    cond = cond[0]  # Select the first channel if condition is multi-channel
                cond_shape = cond.shape
                cond_resized = cond.reshape(cond_shape)

                plt.figure(figsize=(12, 4))

                # Plot condition image
                plt.subplot(1, 3, 1)
                plt.title('Condition Image')
                plt.imshow(cond_resized, cmap='gray')
                plt.axis('off')

                # Plot reference image
                plt.subplot(1, 3, 2)
                plt.title('Reference Image')
                plt.imshow(ref, cmap='gray')
                plt.axis('off')

                # Plot generated image
                plt.subplot(1, 3, 3)
                plt.title('Generated Image')
                plt.imshow(img, cmap='gray')
                plt.axis('off')

                plt.show()