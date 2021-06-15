"""
Glow: Generative Flow with Invertible 1x1 Convolutions
arXiv:1807.03039v2
"""

import os
import shutil
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import load_data
from Evaluator import Evaluator
from Net import Glow_Net
from Model import Glow


if __name__ == '__main__':
    seed = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_norm_clip', default=50.0, type=float)
    parser.add_argument('--trainable', action='store_true')
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()

    train_loader, valid_loader, test_loader = load_data(args.root_folder, batch_size=16)

    net = Glow_Net(width=512, depth=32, n_levels=3, input_dims=(3, 64, 64))

    if args.load:
        net = torch.load(args.load)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    evaluator = Evaluator('./weights/classifier_weight.pth')

    model = Glow(net, optimizer, evaluator)

    if args.trainable:
        image_folder = 'images'
        weight_folder = 'weights/nf'

        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)

        os.makedirs(image_folder)

        if os.path.exists(weight_folder):
            shutil.rmtree(weight_folder)

        os.makedirs(weight_folder)

        model.train(args.epochs, train_loader, valid_loader, warmup_epochs=args.warmup_epochs, grad_norm_clip=args.grad_norm_clip)

    print('test.json')
    model.test(valid_loader)

    print('new_test.json')
    model.test(test_loader)
