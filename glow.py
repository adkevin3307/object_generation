"""
Glow: Generative Flow with Invertible 1x1 Convolutions
arXiv:1807.03039v2
"""

import os
import time
import shutil
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image, make_grid

from utils import load_data
from Evaluator import Evaluator
from Net import Glow


@torch.no_grad()
def generate(model: Glow, condition: torch.Tensor, n_samples: int, z_stds: list) -> torch.Tensor:
    model.eval()

    samples = []
    for z_std in z_stds:
        sample, _ = model.inverse(None, condition, batch_size=n_samples, z_std=z_std)

        samples.append(sample)

    return torch.cat(samples, dim=0)


def train(model: Glow, train_loader: DataLoader, valid_loader: DataLoader, optimizer: optim.Optimizer, evaluator: Evaluator, args):
    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        model.train()

        tic = time.time()
        for i, (image, label) in enumerate(train_loader):
            image = image.to(args.device)
            label = label.to(args.device)

            args.step += args.world_size
            # warmup learning rate
            if epoch <= args.n_epochs_warmup:
                optimizer.param_groups[0]['lr'] = args.lr * min(1, args.step / (len(train_loader) * args.world_size * args.n_epochs_warmup))

            loss = -1.0 * torch.mean(model.log_prob(image, label, bits_per_pixel=True), dim=0)

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)

            optimizer.step()

            et = time.time() - tic
            print(f'Epoch: [{epoch + 1}/{args.start_epoch + args.n_epochs}][{i + 1}/{len(train_loader)}], Time: {(et // 60):.0f}m{(et % 60):.02f}s, Loss: {loss.item():.3f}')

            if (i + 1) % 10 == 0:
                samples = generate(model, label, n_samples=label.shape[0], z_stds=[1.0])

                images = make_grid(samples.cpu(), nrow=4, pad_value=1, normalize=True)
                save_image(images, f'images/generated_sample_{args.step}.png')

                torch.save(model, f'weights/nf/{epoch + 1}.pth')

        model.eval()
        with torch.no_grad():
            accuracy = 0.0

            for i, (_, label) in enumerate(valid_loader):
                label = label.to(args.device)

                samples = generate(model, label, n_samples=label.shape[0], z_stds=[1.0])

                temp_accuracy = evaluator.eval(samples, label)
                accuracy += temp_accuracy

                print(f'\raccuracy: {temp_accuracy:.3f}', end='')

            accuracy /= len(valid_loader)
            print(f'\raccuracy: {accuracy:.3f}')


if __name__ == '__main__':
    seed = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, required=True)
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--n_epochs_warmup', type=int, default=2, help='Number of warmup epochs for linear learning rate annealing.')
    parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--grad_norm_clip', default=50.0, type=float, help='Clip gradients during training.')
    parser.add_argument('--world_size', type=int, default=1, help='Number of nodes for distributed training.')
    parser.add_argument('--trainable', action='store_true')
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()

    args.step = 0  # global step
    args.device = 'cuda' if cuda.is_available() else 'cpu'

    train_loader, valid_loader, test_loader = load_data(args.root_folder, batch_size=16)

    # load model
    model = Glow(width=512, depth=32, n_levels=3, input_dims=(3, 64, 64))

    if args.load:
        model = torch.load(args.load)

    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    evaluator = Evaluator('./weights/classifier_weight.pth')

    if args.trainable:
        image_folder = 'images'
        weight_folder = 'weights/nf'

        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)

        os.makedirs(image_folder)

        if os.path.exists(weight_folder):
            shutil.rmtree(weight_folder)

        os.makedirs(weight_folder)

        train(model, train_loader, valid_loader, optimizer, evaluator, args)

    accuracy = 0.0
    for i, (_, label) in enumerate(valid_loader):
        label = label.to(args.device)

        gen_image = generate(model, label, label.shape[0], z_stds=[1.0])

        temp_accuracy = evaluator.eval(gen_image, label)
        accuracy += temp_accuracy

        print(f'\rAccuracy: {temp_accuracy:.3f}', end='')

    print(f'\rAccuracy: {accuracy:.3f}')

    accuracy = 0.0
    for i, (_, label) in enumerate(test_loader):
        label = label.to(args.device)

        gen_image = generate(model, label, label.shape[0], z_stds=[1.0])

        temp_accuracy = evaluator.eval(gen_image, label)
        accuracy += temp_accuracy

        print(f'\rAccuracy: {temp_accuracy:.3f}', end='')

    print(f'\rAccuracy: {accuracy:.3f}')
