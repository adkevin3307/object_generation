import os
import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from Net import Generator
from Evaluator import Evaluator


def _gen_labels(n: int) -> torch.Tensor:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gen_label = []
    for _ in range(n):
        object_amount = np.random.randint(1, 4, 1)
        temp_gen_label = np.random.choice(range(24), object_amount, replace=False)
        temp_gen_label = one_hot(torch.tensor(temp_gen_label), 24)
        gen_label.append(temp_gen_label.sum(0).view(1, -1))
    gen_label = torch.cat(gen_label, dim=0).type(torch.float).to(device)

    return gen_label


def train(epochs: int, latent_dim: int, model: tuple, optimizer: tuple, criterion: tuple, train_loader: DataLoader, evaluator: Evaluator) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generator, discriminator = model[0], model[1]
    generator_optimizer, discriminator_optimizer = optimizer[0], optimizer[1]
    adversarial_loss, auxiliary_loss = criterion[0], criterion[1]

    generator = generator.to(device)
    discriminator = discriminator.to(device)

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

            # train generator
            generator_optimizer.zero_grad()

            latent = torch.tensor(np.random.normal(0, 1, (batch_size, latent_dim)), dtype=torch.float).to(device)

            gen_label = _gen_labels(batch_size)

            gen_image = generator(latent, label)
            validity, pred_label = discriminator(gen_image)

            temp_g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_label))
            g_loss += temp_g_loss.item()

            temp_g_loss.backward()
            generator_optimizer.step()

            # calculate generator accuracy
            temp_g_accuracy = evaluator.eval(gen_image, label)
            g_accuracy += temp_g_accuracy

            # train discriminator
            discriminator_optimizer.zero_grad()

            # loss for real images
            real_validity, real_label = discriminator(image)
            real_loss = (adversarial_loss(real_validity, valid) + auxiliary_loss(real_label, label)) / 2

            # loss for fake images
            fake_validity, fake_label = discriminator(gen_image.detach())
            fake_loss = (adversarial_loss(fake_validity, fake) + auxiliary_loss(fake_label, gen_label)) / 2

            # total discriminator loss
            temp_d_loss = (real_loss + fake_loss) / 2
            d_loss += temp_d_loss.item()

            temp_d_loss.backward()
            discriminator_optimizer.step()

            # calculate discriminator accuracy
            pred = (torch.cat([real_label, fake_label], dim=0) > 0.5).type(torch.long)
            truth = torch.cat([label, gen_label], dim=0)

            temp_d_accuracy = (torch.sum(torch.logical_and((pred == truth), (truth == 1))) / torch.sum(truth == 1)).item()
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

                save_image(gen_image.data, f'images/{batch_amount}.png', nrow=8, normalize=True)

        g_loss /= len(train_loader)
        d_loss /= len(train_loader)
        g_accuracy /= len(train_loader)
        d_accuracy /= len(train_loader)

        print(f'\r{" " * last_length}', end='')
        print(f'\rEpochs: {(epoch + 1):>{length}} / {epochs}, [{"=" * 20}], ', end='')
        print(f'g_loss: {g_loss:.3f}, d_loss: {d_loss:.3f}, g_accuracy: {g_accuracy:.3f}, d_accuracy: {d_accuracy:.3f}')


def test(latent_dim: int, generator: Generator, test_loader: DataLoader, evaluator: Evaluator) -> None:
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
