import os
import json
import numpy as np
from PIL import Image
from typing import Callable
import torch
from torch.utils.data import Dataset


def get_iCLEVR_data(root_folder, mode):
    with open(os.path.join(root_folder, f'{mode}.json'), 'r') as json_file:
        data = json.load(json_file)

    with open(os.path.join(root_folder, 'objects.json'), 'r') as json_file:
        objects = json.load(json_file)

    image, label = [], []

    if mode == 'train':
        image = list(data.keys())
        label = list(data.values())
    else:
        image = None
        label = data

    for i in range(len(label)):
        for j in range(len(label[i])):
            label[i][j] = objects[label[i][j]]

        index = np.zeros(len(objects))
        index[label[i]] = 1
        label[i] = index

    return (image, label)


class ICLEVRDataset(Dataset):
    def __init__(self, root_folder: str, transform: Callable = None, mode: str = 'train') -> None:
        self.root_folder = root_folder
        self.transform = transform
        self.mode = mode

        self.num_classes = 24
        self.image_list, self.label_list = get_iCLEVR_data(root_folder, mode)

    def __len__(self) -> int:
        """'return the size of dataset"""

        return len(self.label_list)

    def __getitem__(self, index: int) -> tuple:
        image_folder = os.path.join(self.root_folder, 'images')

        if self.image_list == None:
            image = torch.tensor([0])
        else:
            png = Image.open(os.path.join(image_folder, self.image_list[index]))
            png.load()

            image = Image.new('RGB', png.size, (255, 255, 255))
            image.paste(png, mask=png.split()[3])

            if self.transform:
                image = self.transform(image)

        label = self.label_list[index]

        return (image, torch.tensor(label, dtype=torch.float))
