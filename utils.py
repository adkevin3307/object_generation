import argparse
from torch.utils.data import DataLoader
from torchvision import transforms

from ICLEVRDataset import ICLEVRDataset


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--latent', type=int, default=100)
    args = parser.parse_args()

    print('=' * 100)
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('=' * 100)

    return args


def load_data(root_folder: str) -> tuple:
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_set = ICLEVRDataset(root_folder, transforms=train_transforms, mode='train')
    test_set = ICLEVRDataset(root_folder, transforms=test_transforms, mode='test')

    train_loader = DataLoader(train_set, 64, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, 64, shuffle=False, num_workers=8)

    return (train_loader, test_loader)
