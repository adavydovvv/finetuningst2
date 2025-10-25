# experiments/datasets.py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageFolderSimple(Dataset):
    def __init__(self, folder, classes=None, transform=None):
        self.samples = []
        self.labels = []
        self.class_to_idx = {}
        if classes is None:
            classes = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder,d))])
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_dir = os.path.join(folder, cls)
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith(('.jpg','.jpeg','.png')):
                    self.samples.append(os.path.join(cls_dir, fn))
                    self.labels.append(idx)
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)['image']
        label = self.labels[idx]
        return img, label

def get_transforms(image_size):
    train = A.Compose([
        A.RandomResizedCrop((image_size, image_size), scale=(0.8,1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.3),
        A.ColorJitter(0.2,0.2,0.2, p=0.3),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    val = A.Compose([
        A.Resize(image_size, image_size),  # Resize по-прежнему может быть int
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    return train, val

