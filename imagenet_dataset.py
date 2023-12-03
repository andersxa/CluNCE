import glob
import json
from PIL import Image
from torch.utils.data import Dataset
import os

class ImageNet(Dataset):
    def __init__(self, root, split, transform=None):
        self.images = []
        self.classes = []
        self.transform = transform
        self.syn_to_class = {}
        with open(f"{root}/imagenet_class_index.json", "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        with open(f"{root}/ILSVRC2012_val_labels.json", "rb") as f:
            self.val_to_syn = json.load(f)
        images_dir = f"{root}/ILSVRC/Data/CLS-LOC/{split}"

        for folder in glob.glob(f"{images_dir}/*"):
            syn_id = os.path.basename(folder)
            if split == "train":
                image_class = self.syn_to_class[syn_id]
                for img in glob.glob(f"{folder}/*"):
                    self.images.append(img)
                    self.classes.append(image_class)
            elif split == "val":
                syn_id = self.val_to_syn[syn_id]
                image_class = self.syn_to_class[syn_id]
                self.images.append(folder)
                self.classes.append(image_class)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        with open(path, "rb") as f:
            img = Image.open(f)
            x = img.convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.classes[idx]