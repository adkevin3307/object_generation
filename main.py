import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import parse, load_data
from Net import ACGAN_Generator, ACGAN_Discriminator, WGAN_Generator, WGAN_Discriminator
from Evaluator import Evaluator
from Model import ACGAN, WGAN


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(0)

    args = parse()

    train_loader, valid_loader, test_loader = load_data(args.root_folder)

    evaluator = Evaluator('./weights/classifier_weight.pth')

    if args.net == 'ACGAN':
        generator = ACGAN_Generator(args.latent_dim, image_size=64)
        discriminator = ACGAN_Discriminator(image_size=64)

        generator_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        adversarial_criterion = nn.BCELoss()
        auxiliary_criterion = nn.BCELoss()

        model = ACGAN(
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            adversarial_criterion,
            auxiliary_criterion,
            evaluator,
            args.latent_dim,
            num_classes=24
        )
    elif args.net == 'WGAN':
        generator = WGAN_Generator(args.latent_dim, image_size=64)
        discriminator = WGAN_Discriminator(image_size=64)

        generator_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        model = WGAN(
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            evaluator,
            args.latent_dim,
            num_classes=24
        )
    else:
        raise NotImplementedError('net architecture not implemented')

    model.load(args.load_generator, args.load_discriminator)

    if args.trainable:
        image_folder = 'images'
        generator_folder = 'weights/generator'
        discriminator_folder = 'weights/discriminator'

        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)

        if os.path.exists(generator_folder):
            shutil.rmtree(generator_folder)

        if os.path.exists(discriminator_folder):
            shutil.rmtree(discriminator_folder)

        os.makedirs(image_folder)
        os.makedirs(generator_folder)
        os.makedirs(discriminator_folder)

        model.train(args.epochs, train_loader, valid_loader, verbose=False)

        model.save('weights/generator/final.pth', 'weights/discriminator/final.pth')

    print('test.json')
    model.test(valid_loader)

    print('new_test.json')
    model.test(test_loader)
