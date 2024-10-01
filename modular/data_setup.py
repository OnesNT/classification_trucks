import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

class TruckDataset(Dataset):
    classes = ['truck_1', 'truck_2']

    def __init__(self, root_dir=None, img=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img = img
        self.img_paths = []
        self.labels = []

        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.img_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

