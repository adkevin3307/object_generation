import os
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms

from ICLEVRDataset import ICLEVRDataset


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, required=True)
    parser.add_argument('--net', type=str, default='WGAN', choices=['WGAN', 'ACGAN'])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--load_generator', type=str, default=None)
    parser.add_argument('--load_discriminator', type=str, default=None)
    parser.add_argument('--trainable', action='store_true', default=False)
    args = parser.parse_args()

    print('=' * 100)
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('=' * 100)

    return args


def load_data(root_folder: str) -> tuple:
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_set = ICLEVRDataset(root_folder, transform=train_transform, mode='train')
    valid_set = ICLEVRDataset(root_folder, transform=test_transform, mode='test')
    test_set = ICLEVRDataset(root_folder, transform=test_transform, mode='new_test')

    num_workers = len(os.sched_getaffinity(0))
    train_loader = DataLoader(train_set, 64, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, 64, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, 64, shuffle=False, num_workers=num_workers)

    return (train_loader, valid_loader, test_loader)
