import os
import numpy as np
from typing import Any
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from Evaluator import Evaluator


def _gen_latent(size: int, latent_dim: int) -> torch.Tensor:
    return torch.tensor(np.random.normal(0, 1, (size, latent_dim)), dtype=torch.float)


def _gen_label(size: int, max_object_amount: int = 4, class_amount: int = 24) -> torch.Tensor:
    gen_label = []

    for _ in range(size):
        object_amount = np.random.randint(1, max_object_amount, 1)

        temp_gen_label = np.random.choice(range(class_amount), object_amount, replace=False)
        temp_gen_label = one_hot(torch.tensor(temp_gen_label), class_amount)

        gen_label.append(torch.sum(temp_gen_label, dim=0).view(1, -1))

    return torch.cat(gen_label, dim=0).type(torch.float)


def _save(n: int, latent_dim: int, generator: Any, discriminator: Any) -> None:
    size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.save(generator, f'weights/generator/generator_{n}.pth')
    torch.save(discriminator, f'weights/discriminator/discriminator_{n}.pth')

    latent = _gen_latent(size, latent_dim).to(device)
    label = _gen_label(size).to(device)

    gen_image = generator(latent, label)

    save_image(gen_image.data, f'images/{n}.png', nrow=int(size ** 0.5), normalize=True)


def train(epochs: int, latent_dim: int, model: tuple, optimizer: tuple, criterion: tuple, train_loader: DataLoader, evaluator: Evaluator, valid_loader: DataLoader = None, verbose: bool = True) -> None:
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

        last_length = 0
        epoch_length = len(str(epochs))

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

            latent = _gen_latent(batch_size, latent_dim).to(device)
            gen_label = _gen_label(batch_size).to(device)

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

                fake_image = generator(latent, label)

                fake_validity = discriminator(fake_image, label)

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

            message = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}, [{progress_bar:<20}] {current_progress:>6.2f}%'

            if verbose:
                message += ', '
                message += f'g_loss: {temp_g_loss.item():.3f}, d_loss: {temp_d_loss.item():.3f}, '
                message += f'g_accuracy: {temp_g_accuracy:.3f}, d_accuracy: {temp_d_accuracy:.3f}'

            last_length = len(message) + 1

            print(f'\r{message}', end='')

        if (epoch + 1) % 5 == 0:
            _save(epoch + 1, latent_dim, generator, discriminator)

        g_loss /= len(train_loader)
        d_loss /= len(train_loader)
        g_accuracy /= len(train_loader)
        d_accuracy /= len(train_loader)

        print(f'\r{" " * last_length}', end='')
        print(f'\rEpochs: {(epoch + 1):>{epoch_length}} / {epochs}, [{"=" * 20}], ', end='')
        print(f'g_loss: {g_loss:.3f}, d_loss: {d_loss:.3f}, g_accuracy: {g_accuracy:.3f}, d_accuracy: {d_accuracy:.3f}', end=', ' if valid_loader else '\n')

        if valid_loader:
            test(latent_dim, generator, valid_loader, evaluator, is_test=False)


def test(latent_dim: int, generator: Any, test_loader: DataLoader, evaluator: Evaluator, is_test: bool = True) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generator.eval()

    accuracy = 0.0
    last_length = 0

    with torch.no_grad():
        for i, (_, label) in enumerate(test_loader):
            label = label.to(device)

            batch_size = label.shape[0]

            latent = _gen_latent(batch_size, latent_dim).to(device)

            gen_image = generator(latent, label)

            temp_accuracy = evaluator.eval(gen_image, label)
            accuracy += temp_accuracy

            if is_test:
                current_progress = (i + 1) / len(test_loader) * 100
                progress_bar = '=' * int((i + 1) * (20 / len(test_loader)))

                print(f'\r{" " * last_length}', end='')

                message = f'Test: [{progress_bar:<20}] {current_progress:>6.2f}%, accuracy: {temp_accuracy:.3f}'
                last_length = len(message) + 1

                print(f'\r{message}', end='')

    accuracy /= len(test_loader)

    if is_test:
        print(f'\r{" " * last_length}', end='')
        print(f'\rTest: [{"=" * 20}], ', end='')

    print(f'accuracy: {accuracy:.3f}')
