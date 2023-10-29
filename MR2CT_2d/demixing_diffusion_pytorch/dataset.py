import torch

from torch.utils import data
from pathlib import Path
from torchvision import transforms

import numpy as np

class DatasetPaired_Aug(data.Dataset):
    def __init__(self, folder, image_size, keywords=['MR_x', 'CT_y'], exts=['npy']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.keywords = keywords
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            # Add any other transformations that should be applied to both images
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Add more transformations as needed
        ])

    def __len__(self):
        # Ensure that the number of images is even
        return len(self.paths) // 2

    def __getitem__(self, index):
        # Load two images and ensure that they are paired correctly
        # the path 2 is to replace the keyword[0] in path1 to keyword[2]
        path1 = self.paths[index]
        path2 = Path(str(path1).replace(self.keywords[0], self.keywords[1]))
        
        img1 = np.load(path1).astype(np.float32)
        img2 = np.load(path2).astype(np.float32)
        
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        
        # Apply the same transformations to both images
        # img1 = self.transform(img1)
        # img2 = self.transform(img2)
        
        return img1, img2

class DatasetPaired(data.Dataset):
    def __init__(self, folder, image_size, keywords=['MR_x', 'CT_y'], exts=['npy']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.keywords = keywords
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            # Add any other transformations that should be applied to both images
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Add more transformations as needed
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Load two images and ensure that they are paired correctly
        path1 = self.paths[index]
        path2 = Path(str(path1).replace(self.keywords[0], self.keywords[1]))

        img1 = np.load(path1).astype(np.float32)
        img2 = np.load(path2).astype(np.float32)
        
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        
        return img1, img2
    
# class Dataset_Aug1(data.Dataset):
#     def __init__(self, folder, image_size, exts = ['npy']):
#         super().__init__()
#         self.folder = folder
#         self.image_size = image_size
#         self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

#         self.transform = transforms.Compose([
#             # transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
#             # transforms.RandomCrop(image_size),
#             # make sure the channel first
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             # transforms.ToTensor(),
#             # transforms.Lambda(lambda t: (t * 2) - 1)
#         ])

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, index):
#         path = self.paths[index]
#         img = np.load(path).astype(np.float32)
#         # convert img from double to float
#         img = torch.from_numpy(img)
#         # img = Image.open(path)
#         # img = img.convert('RGB')
#         # return img
#         return self.transform(img)

# class Dataset(data.Dataset):
#     def __init__(self, folder, image_size, exts=['npy']):
#         super().__init__()
#         self.folder = folder
#         self.image_size = image_size
#         self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

#         self.transform = transforms.Compose([
#             # transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
#             # transforms.CenterCrop(image_size),
#             # transforms.ToTensor(),
#             # transforms.Lambda(lambda t: (t * 2) - 1)
#         ])

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, index):
#         path = self.paths[index]
#         img = np.load(path).astype(np.float32)
#         # convert img to pytorch tensor
#         img = torch.from_numpy(img)
#         # img = Image.open(path)
#         # img = img.convert('RGB')
#         return img
#         # return self.transform(img)