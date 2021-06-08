import os
import numpy as np
from typing import Any
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from Evaluator import Evaluator


def train(epochs: int, latent_dim: int, model: tuple, optimizer: tuple, criterion: tuple, train_loader: DataLoader, evaluator: Evaluator) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generator, discriminator = model[0], model[1]
    generator_optimizer, discriminator_optimizer = optimizer[0], optimizer[1]
    adversarial_criterion, auxiliary_criterion = criterion[0], criterion[1]

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    generator_name = generator._get_name()
    discriminator_name = discriminator._get_name()

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        length = len(str(epochs))
        last_length = 0

        g_loss = 0.0
        d_loss = 0.0
        g_accuracy = 0.0
        d_accuracy = 0.0

        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)

            batch_size = image.shape[0]

            valid = torch.tensor(np.ones((batch_size, 1)), dtype=torch.float).to(device)
            valid.requires_grad = False

            fake = torch.tensor(np.zeros((batch_size, 1)), dtype=torch.float).to(device)
            fake.requires_grad = False

            latent = torch.tensor(np.random.normal(0, 1, (batch_size, latent_dim)), dtype=torch.float).to(device)

            gen_label = []
            for _ in range(batch_size):
                object_amount = np.random.randint(1, 4, 1)
                temp_gen_label = np.random.choice(range(24), object_amount, replace=False)
                temp_gen_label = one_hot(torch.tensor(temp_gen_label), 24)
                gen_label.append(temp_gen_label.sum(0).view(1, -1))
            gen_label = torch.cat(gen_label, dim=0).type(torch.float).to(device)

            generator_optimizer.zero_grad()

            gen_image = generator(latent, label)

            if generator_name == 'ACGAN_Generator':
                validity, pred_label = discriminator(gen_image)

                temp_g_loss = 0.5 * (adversarial_criterion(validity, valid) + auxiliary_criterion(pred_label, gen_label))
                g_loss += temp_g_loss.item()

                temp_g_loss.backward()
                generator_optimizer.step()

            elif generator_name == 'WGAN_Generator':
                temp_g_loss = torch.zeros(1)

                if (i + 1) % 5 == 0:
                    validity = discriminator(gen_image, label)

                    temp_g_loss = -1.0 * torch.mean(validity)
                    g_loss += temp_g_loss.item()

                    temp_g_loss.backward()
                    generator_optimizer.step()
            else:
                raise NotImplementedError('generator loss and accuracy not implemented')

            temp_g_accuracy = evaluator.eval(gen_image, label)
            g_accuracy += temp_g_accuracy

            discriminator_optimizer.zero_grad()

            if discriminator_name == 'ACGAN_Discriminator':
                real_validity, real_label = discriminator(image)
                real_loss = (adversarial_criterion(real_validity, valid) + auxiliary_criterion(real_label, label)) / 2

                fake_validity, fake_label = discriminator(gen_image.detach())
                fake_loss = (adversarial_criterion(fake_validity, fake) + auxiliary_criterion(fake_label, gen_label)) / 2

                temp_d_loss = (real_loss + fake_loss) / 2
                d_loss += temp_d_loss.item()

                temp_d_loss.backward()
                discriminator_optimizer.step()

                pred = (torch.cat([real_label, fake_label], dim=0) > 0.5).type(torch.long)
                truth = torch.cat([label, gen_label], dim=0)

                temp_d_accuracy = (torch.sum(torch.logical_and((pred == truth), (truth == 1))) / torch.sum(truth == 1)).item()

            elif discriminator_name == 'WGAN_Discriminator':
                real_image = torch.clone(image).to(device)
                real_image.requires_grad = True

                real_validity = discriminator(real_image, label)

                if torch.rand(1) > 0.5:
                    fake_image = real_image
                    fake_label = gen_label
                else:
                    fake_image = generator(latent, label)
                    fake_label = label

                fake_validity = discriminator(fake_image, fake_label)

                real_grad_out = torch.ones(real_validity.shape).to(device)
                real_grad_out.requires_grad = False

                real_grad = torch.autograd.grad(real_validity, real_image, real_grad_out, retain_graph=True, create_graph=True, only_inputs=True)[0]
                real_grad_norm = torch.sum(real_grad.view(real_grad.shape[0], -1) ** 2, dim=1) ** 3

                fake_grad_out = torch.ones(fake_validity.shape).to(device)
                fake_grad_out.requires_grad = False

                fake_grad = torch.autograd.grad(fake_validity, fake_image, fake_grad_out, retain_graph=True, create_graph=True, only_inputs=True)[0]
                fake_grad_norm = torch.sum(fake_grad.view(fake_grad.shape[0], -1) ** 2, dim=1) ** 3

                temp_d_loss = -1.0 * torch.mean(real_validity) + torch.mean(fake_validity) + torch.mean(real_grad_norm + fake_grad_norm)
                d_loss += temp_d_loss.item()

                temp_d_loss.backward()
                discriminator_optimizer.step()

                pred = (torch.cat([real_validity, fake_validity], dim=0) > 0.5).type(torch.long)
                truth = torch.cat([torch.ones(real_validity.shape), torch.zeros(fake_validity.shape)], dim=0).to(device)

                temp_d_accuracy = torch.sum(pred == truth).item() / truth.shape[0]
            else:
                raise NotImplementedError('discriminator loss and accuracy not implemented')

            d_accuracy += temp_d_accuracy

            # progress bar
            current_progress = (i + 1) / len(train_loader) * 100
            progress_bar = '=' * int((i + 1) * (20 / len(train_loader)))

            print(f'\r{" " * last_length}', end='')

            message = f'Epochs: {(epoch + 1):>{length}} / {epochs}, [{progress_bar:<20}] {current_progress:>6.2f}%, '
            message += f'g_loss: {temp_g_loss.item():.3f}, d_loss: {temp_d_loss.item():.3f}, '
            message += f'g_accuracy: {temp_g_accuracy:.3f}, d_accuracy: {temp_d_accuracy:.3f}'
            last_length = len(message) + 1

            print(f'\r{message}', end='')

            batch_amount = epoch * len(train_loader) + i
            if batch_amount % 500 == 0:
                os.makedirs('weights/generator', exist_ok=True)
                os.makedirs('weights/discriminator', exist_ok=True)

                torch.save(generator, f'weights/generator/generator_{batch_amount}.pth')
                torch.save(discriminator, f'weights/discriminator/discriminator_{batch_amount}.pth')

                os.makedirs('images', exist_ok=True)

                save_image(gen_image.data, f'images/{batch_amount}.png', nrow=8, normalize=True)

        g_loss /= len(train_loader)
        d_loss /= len(train_loader)
        g_accuracy /= len(train_loader)
        d_accuracy /= len(train_loader)

        print(f'\r{" " * last_length}', end='')
        print(f'\rEpochs: {(epoch + 1):>{length}} / {epochs}, [{"=" * 20}], ', end='')
        print(f'g_loss: {g_loss:.3f}, d_loss: {d_loss:.3f}, g_accuracy: {g_accuracy:.3f}, d_accuracy: {d_accuracy:.3f}')


def test(latent_dim: int, generator: Any, test_loader: DataLoader, evaluator: Evaluator) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    accuracy = 0.0
    last_length = 0

    for i, (_, label) in enumerate(test_loader):
        label = label.to(device)

        batch_size = label.shape[0]

        latent = torch.tensor(np.random.normal(0, 1, (batch_size, latent_dim)), dtype=torch.float).to(device)

        gen_image = generator(latent, label)

        temp_accuracy = evaluator.eval(gen_image, label)
        accuracy += temp_accuracy

        current_progress = (i + 1) / len(test_loader) * 100
        progress_bar = '=' * int((i + 1) * (20 / len(test_loader)))

        print(f'\r{" " * last_length}', end='')

        message = f'Test: [{progress_bar:<20}] {current_progress:>6.2f}%, accuracy: {temp_accuracy:.3f}'
        last_length = len(message) + 1

        print(f'\r{message}', end='')

    print(f'\r{" " * last_length}', end='')
    print(f'\rTest: [{"=" * 20}], accuracy: {accuracy:.3f}')
