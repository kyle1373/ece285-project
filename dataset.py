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
    def __init__(self, image_list, cond_type='None', cols=[0,0], rows=[0,0]):
        self.image_list = image_list
        self.transform_image = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),  # Adjust size as needed
            transforms.ToTensor(),  # Converts to tensor and scales pixel values to [0, 1]
        ])
        self.cond_type = cond_type
        
        self.start_col = cols[0]
        self.end_col = cols[1]
        
        self.start_row = rows[0]
        self.end_row = rows[1]
        
#         # Define transformations including data augmentation
#         transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.RandomRotation(10),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name)
        
        if self.transform_image:
            image = self.transform_image(image)
        
        C, H, W = image.shape
        if self.cond_type == 'Half':
            condition = image[:, :H//2, :]  # Extracting the first half rows as condition
        elif self.cond_type == 'Col':
            condition = image[:, :, self.start_col:self.end_col]  # Extracting the given columns as condition
        elif self.cond_type == 'Row':
            condition = image[:, self.start_row:self.end_row, :]  # Extracting the given rows as condition
        else:
            condition = image[:, self.start_row:self.end_row, self.start_col:self.end_col]  # Extracting the given rows and columns as condition

        return image, condition