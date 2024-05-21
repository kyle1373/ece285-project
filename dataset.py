import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
                
def get_full_list(root_dir):
    data_list = sorted(
        os.path.join(root_dir, img_name) for img_name in
        filter(lambda x: x.endswith('.png'), os.listdir(root_dir))
    )
    return data_list

class ChineseCharacterDataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
        self.transform_image = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),  # Adjust size as needed
            transforms.ToTensor(),  # Converts to tensor and scales pixel values to [0, 1]
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name)
        
        if self.transform_image:
            image = self.transform_image(image)

        return image