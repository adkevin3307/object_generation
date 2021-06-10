import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import parse, load_data
from Net import ACGAN_Generator, ACGAN_Discriminator, WGAN_Generator, WGAN_Discriminator
from Evaluator import Evaluator
from Model import train, test


def init_weights(m):
    name = m.__class__.__name__

    if 'Conv' in name:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm2d' in name:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    args = parse()

    train_loader, valid_loader, test_loader = load_data(args.root_folder)

    if args.net == 'ACGAN':
        generator = ACGAN_Generator(args.latent, image_size=64)
        discriminator = ACGAN_Discriminator(image_size=64)
    elif args.net == 'WGAN':
        generator = WGAN_Generator(args.latent, image_size=64)
        discriminator = WGAN_Discriminator(image_size=64)
    else:
        raise NotImplementedError('net architecture not implemented')

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    generator_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    adversarial_criterion = nn.BCELoss()
    auxiliary_criterion = nn.BCELoss()

    evaluator = Evaluator('./weights/classifier_weight.pth')

    if args.trainable:
        train(
            epochs=args.epochs,
            latent_dim=args.latent,
            model=(generator, discriminator),
            optimizer=(generator_optimizer, discriminator_optimizer),
            criterion=(adversarial_criterion, auxiliary_criterion),
            train_loader=train_loader,
            evaluator=evaluator,
            valid_loader=valid_loader,
            verbose=False
        )

        torch.save(generator, 'weights/generator/generator_final.pth')
        torch.save(discriminator, 'weights/discriminator/discriminator_final.pth')

    if args.load:
        generator = torch.load(args.load)

    test(args.latent, generator, test_loader, evaluator)
