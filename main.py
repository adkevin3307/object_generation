import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import parse, load_data
from Net import ACGAN_Generator, ACGAN_Discriminator, WGAN_Generator, WGAN_Discriminator
from Evaluator import Evaluator
from Model import ACGAN, WGAN


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
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
        os.makedirs('images', exist_ok=True)
        os.makedirs('weights/generator', exist_ok=True)
        os.makedirs('weights/discriminator', exist_ok=True)

        model.train(args.epochs, train_loader, valid_loader, verbose=False)

        model.save('weights/generator/final.pth', 'weights/discriminator/final.pth')

    print('test.json')
    model.test(valid_loader)

    print('new_test.json')
    model.test(test_loader)
