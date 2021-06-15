import os
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms

from ICLEVRDataset import ICLEVRDataset


def parse_gan() -> argparse.Namespace:
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


def parse_nf() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_norm_clip', default=50.0, type=float)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--trainable', action='store_true')
    args = parser.parse_args()

    print('=' * 100)
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('=' * 100)

    return args


def load_data(root_folder: str, batch_size: int) -> tuple:
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
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=num_workers)

    return (train_loader, valid_loader, test_loader)
