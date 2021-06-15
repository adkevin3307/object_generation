import time
import numpy as np
from typing import Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from Evaluator import Evaluator
from Net import ACGAN_Generator, ACGAN_Discriminator, WGAN_Generator, WGAN_Discriminator, Glow_Net


class BaseModel:
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self) -> None:
        raise NotImplementedError('train not implemented')

    def test(self) -> None:
        raise NotImplementedError('test not implemented')

    def load(self) -> None:
        raise NotImplementedError('load not implemented')

    def save(self) -> None:
        raise NotImplementedError('save not implemented')


class ACGAN(BaseModel):
    def __init__(
        self,
        generator: ACGAN_Generator,
        discriminator: ACGAN_Discriminator,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        adversarial_criterion: Callable,
        auxiliary_criterion: Callable,
        evaluator: Evaluator,
        latent_dim: int,
        num_classes: int
    ) -> None:
        super(ACGAN, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.adversarial_criterion = adversarial_criterion
        self.auxiliary_criterion = auxiliary_criterion
        self.evaluator = evaluator
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.generator.apply(self._init_weight)
        self.discriminator.apply(self._init_weight)

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

    def _init_weight(self, m) -> None:
        name = m.__class__.__name__

        if 'Conv' in name:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif 'BatchNorm2d' in name:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def _gen_latent(self, size: int) -> torch.Tensor:
        return torch.tensor(np.random.normal(0, 1, (size, self.latent_dim)), dtype=torch.float)

    def _gen_label(self, size: int, max_object_amount: int = 4) -> torch.Tensor:
        gen_label = []

        for _ in range(size):
            object_amount = np.random.randint(1, max_object_amount, 1)

            temp_gen_label = np.random.choice(range(self.num_classes), object_amount, replace=False)
            temp_gen_label = one_hot(torch.tensor(temp_gen_label), self.num_classes)

            gen_label.append(torch.sum(temp_gen_label, dim=0).view(1, -1))

        return torch.cat(gen_label, dim=0).type(torch.float)

    def _save_image(self, image_name: str) -> None:
        size = 64

        latent = self._gen_latent(size).to(self.device)
        label = self._gen_label(size).to(self.device)

        gen_image = self.generator(latent, label)

        save_image(gen_image.data, image_name, nrow=int(size ** 0.5), normalize=True)

    def train(self, epochs: int, train_loader: DataLoader, valid_loader: DataLoader = None, verbose: bool = True) -> None:
        for epoch in range(epochs):
            self.generator.train()
            self.discriminator.train()

            last_length = 0
            epoch_length = len(str(epochs))

            g_loss = 0.0
            d_loss = 0.0
            g_accuracy = 0.0
            d_accuracy = 0.0

            for i, (image, label) in enumerate(train_loader):
                image = image.to(self.device)
                label = label.to(self.device)

                batch_size = image.shape[0]

                valid = torch.tensor(np.ones((batch_size, 1)), dtype=torch.float).to(self.device)
                valid.requires_grad = False

                fake = torch.tensor(np.zeros((batch_size, 1)), dtype=torch.float).to(self.device)
                fake.requires_grad = False

                latent = self._gen_latent(batch_size).to(self.device)
                gen_label = self._gen_label(batch_size).to(self.device)

                self.generator_optimizer.zero_grad()

                gen_image = self.generator(latent, label)

                validity, pred_label = self.discriminator(gen_image)

                temp_g_loss = 0.5 * (self.adversarial_criterion(validity, valid) + self.auxiliary_criterion(pred_label, gen_label))
                g_loss += temp_g_loss.item()

                temp_g_loss.backward()
                self.generator_optimizer.step()

                temp_g_accuracy = self.evaluator.eval(gen_image, label)
                g_accuracy += temp_g_accuracy

                self.discriminator_optimizer.zero_grad()

                real_validity, real_label = self.discriminator(image)
                real_loss = (self.adversarial_criterion(real_validity, valid) + self.auxiliary_criterion(real_label, label)) / 2

                fake_validity, fake_label = self.discriminator(gen_image.detach())
                fake_loss = (self.adversarial_criterion(fake_validity, fake) + self.auxiliary_criterion(fake_label, gen_label)) / 2

                temp_d_loss = (real_loss + fake_loss) / 2
                d_loss += temp_d_loss.item()

                temp_d_loss.backward()
                self.discriminator_optimizer.step()

                pred = (torch.cat([real_label, fake_label], dim=0) > 0.5).type(torch.long)
                truth = torch.cat([label, gen_label], dim=0)

                temp_d_accuracy = (torch.sum(torch.logical_and((pred == truth), (truth == 1))) / torch.sum(truth == 1)).item()
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
                self.save(f'weights/generator/{epoch + 1}.pth', f'weights/discriminator/{epoch + 1}.pth')
                self._save_image(f'images/{epoch + 1}.png')

            g_loss /= len(train_loader)
            d_loss /= len(train_loader)
            g_accuracy /= len(train_loader)
            d_accuracy /= len(train_loader)

            print(f'\r{" " * last_length}', end='')
            print(f'\rEpochs: {(epoch + 1):>{epoch_length}} / {epochs}, [{"=" * 20}], ', end='')
            print(f'g_loss: {g_loss:.3f}, d_loss: {d_loss:.3f}, g_accuracy: {g_accuracy:.3f}, d_accuracy: {d_accuracy:.3f}', end=', ' if valid_loader else '\n')

            if valid_loader:
                self.test(valid_loader, is_test=False)

    def test(self, test_loader: DataLoader, is_test: bool = True) -> None:
        self.generator.eval()

        accuracy = 0.0
        last_length = 0

        with torch.no_grad():
            for i, (_, label) in enumerate(test_loader):
                label = label.to(self.device)

                batch_size = label.shape[0]

                latent = self._gen_latent(batch_size).to(self.device)

                gen_image = self.generator(latent, label)

                temp_accuracy = self.evaluator.eval(gen_image, label)
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

    def save(self, generator_name: str, discriminator_name: str) -> None:
        torch.save(self.generator, generator_name)
        torch.save(self.discriminator, discriminator_name)

    def load(self, generator_name: str, discriminator_name: str) -> None:
        if generator_name:
            self.generator = torch.load(generator_name)
            self.generator_optimizer.param_groups[0]['params'] = self.generator.parameters()

        if discriminator_name:
            self.discriminator = torch.load(discriminator_name)
            self.discriminator_optimizer.param_groups[0]['params'] = self.discriminator.parameters()


class WGAN(BaseModel):
    def __init__(
        self,
        generator: WGAN_Generator,
        discriminator: WGAN_Discriminator,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        evaluator: Evaluator,
        latent_dim: int,
        num_classes: int
    ) -> None:
        super(WGAN, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.evaluator = evaluator
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.generator.apply(self._init_weight)
        self.discriminator.apply(self._init_weight)

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

    def _init_weight(self, m) -> None:
        name = m.__class__.__name__

        if 'Conv' in name:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif 'BatchNorm2d' in name:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def _gen_latent(self, size: int) -> torch.Tensor:
        return torch.tensor(np.random.normal(0, 1, (size, self.latent_dim)), dtype=torch.float)

    def _gen_label(self, size: int, max_object_amount: int = 4) -> torch.Tensor:
        gen_label = []

        for _ in range(size):
            object_amount = np.random.randint(1, max_object_amount, 1)

            temp_gen_label = np.random.choice(range(self.num_classes), object_amount, replace=False)
            temp_gen_label = one_hot(torch.tensor(temp_gen_label), self.num_classes)

            gen_label.append(torch.sum(temp_gen_label, dim=0).view(1, -1))

        return torch.cat(gen_label, dim=0).type(torch.float)

    def _save_image(self, image_name: str) -> None:
        size = 64

        latent = self._gen_latent(size).to(self.device)
        label = self._gen_label(size).to(self.device)

        gen_image = self.generator(latent, label)

        save_image(gen_image.data, image_name, nrow=int(size ** 0.5), normalize=True)

    def train(self, epochs: int, train_loader: DataLoader, valid_loader: DataLoader = None, verbose: bool = True) -> None:
        for epoch in range(epochs):
            self.generator.train()
            self.discriminator.train()

            last_length = 0
            epoch_length = len(str(epochs))

            g_loss = 0.0
            d_loss = 0.0
            g_accuracy = 0.0
            d_accuracy = 0.0

            for i, (image, label) in enumerate(train_loader):
                image = image.to(self.device)
                label = label.to(self.device)

                batch_size = image.shape[0]

                latent = self._gen_latent(batch_size).to(self.device)

                self.generator_optimizer.zero_grad()

                gen_image = self.generator(latent, label)

                temp_g_loss = torch.zeros(1)

                if (i + 1) % 5 == 0:
                    validity = self.discriminator(gen_image, label)

                    temp_g_loss = -1.0 * torch.mean(validity)
                    g_loss += temp_g_loss.item()

                    temp_g_loss.backward()
                    self.generator_optimizer.step()

                temp_g_accuracy = self.evaluator.eval(gen_image, label)
                g_accuracy += temp_g_accuracy

                self.discriminator_optimizer.zero_grad()

                real_image = torch.clone(image).to(self.device)
                real_image.requires_grad = True

                real_validity = self.discriminator(real_image, label)

                fake_image = self.generator(latent, label)

                fake_validity = self.discriminator(fake_image, label)

                real_grad_out = torch.ones(real_validity.shape).to(self.device)
                real_grad_out.requires_grad = False

                real_grad = torch.autograd.grad(real_validity, real_image, real_grad_out, retain_graph=True, create_graph=True, only_inputs=True)[0]
                real_grad_norm = torch.sum(real_grad.view(real_grad.shape[0], -1) ** 2, dim=1) ** 3

                fake_grad_out = torch.ones(fake_validity.shape).to(self.device)
                fake_grad_out.requires_grad = False

                fake_grad = torch.autograd.grad(fake_validity, fake_image, fake_grad_out, retain_graph=True, create_graph=True, only_inputs=True)[0]
                fake_grad_norm = torch.sum(fake_grad.view(fake_grad.shape[0], -1) ** 2, dim=1) ** 3

                temp_d_loss = -1.0 * torch.mean(real_validity) + torch.mean(fake_validity) + torch.mean(real_grad_norm + fake_grad_norm)
                d_loss += temp_d_loss.item()

                temp_d_loss.backward()
                self.discriminator_optimizer.step()

                pred = (torch.cat([real_validity, fake_validity], dim=0) > 0.5).type(torch.long)
                truth = torch.cat([torch.ones(real_validity.shape), torch.zeros(fake_validity.shape)], dim=0).to(self.device)

                temp_d_accuracy = torch.sum(pred == truth).item() / truth.shape[0]
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
                self.save(f'weights/generator/{epoch + 1}.pth', f'weights/discriminator/{epoch + 1}.pth')
                self._save_image(f'images/{epoch + 1}.png')

            g_loss /= len(train_loader)
            d_loss /= len(train_loader)
            g_accuracy /= len(train_loader)
            d_accuracy /= len(train_loader)

            print(f'\r{" " * last_length}', end='')
            print(f'\rEpochs: {(epoch + 1):>{epoch_length}} / {epochs}, [{"=" * 20}], ', end='')
            print(f'g_loss: {g_loss:.3f}, d_loss: {d_loss:.3f}, g_accuracy: {g_accuracy:.3f}, d_accuracy: {d_accuracy:.3f}', end=', ' if valid_loader else '\n')

            if valid_loader:
                self.test(valid_loader, is_test=False)

    def test(self, test_loader: DataLoader, is_test: bool = True) -> None:
        self.generator.eval()

        accuracy = 0.0
        last_length = 0

        with torch.no_grad():
            for i, (_, label) in enumerate(test_loader):
                label = label.to(self.device)

                batch_size = label.shape[0]

                latent = self._gen_latent(batch_size).to(self.device)

                gen_image = self.generator(latent, label)

                temp_accuracy = self.evaluator.eval(gen_image, label)
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

    def save(self, generator_name: str, discriminator_name: str) -> None:
        torch.save(self.generator, generator_name)
        torch.save(self.discriminator, discriminator_name)

    def load(self, generator_name: str, discriminator_name: str) -> None:
        if generator_name:
            self.generator = torch.load(generator_name)
            self.generator_optimizer.param_groups[0]['params'] = self.generator.parameters()

        if discriminator_name:
            self.discriminator = torch.load(discriminator_name)
            self.discriminator_optimizer.param_groups[0]['params'] = self.discriminator.parameters()


class Glow(BaseModel):
    def __init__(self, net: Glow_Net, optimizer: optim.Optimizer, evaluator: Evaluator) -> None:
        super(Glow, self).__init__()

        self.net = net
        self.optimizer = optimizer
        self.evaluator = evaluator

        self.net = self.net.to(self.device)

    @torch.no_grad()
    def _generate(self, condition: torch.Tensor, n_samples: int, z_stds: list) -> torch.Tensor:
        self.net.eval()

        samples = []
        for z_std in z_stds:
            sample, _ = self.net.inverse(None, condition, batch_size=n_samples, z_std=z_std)

            samples.append(sample)

        return torch.cat(samples, dim=0)

    def train(self, epochs: int, train_loader: DataLoader, valid_loader: DataLoader = None, warmup_epochs: int = 0, grad_norm_clip: float = 0.0) -> None:
        step = 0

        for epoch in range(epochs):
            self.net.train()

            tic = time.time()
            for i, (image, label) in enumerate(train_loader):
                image = image.to(self.device)
                label = label.to(self.device)

                step += 1

                # warmup learning rate
                if epoch <= warmup_epochs:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.optimizer.param_groups[0]['lr'] = lr * min(1, step / (len(train_loader) * warmup_epochs))

                self.optimizer.zero_grad()

                loss = -1.0 * torch.mean(self.net.log_prob(image, label, bits_per_pixel=True), dim=0)

                loss.backward()

                nn.utils.clip_grad_norm_(self.net.parameters(), grad_norm_clip)

                self.optimizer.step()

                et = time.time() - tic
                print(f'Epoch: [{epoch + 1}/{epochs}][{i + 1}/{len(train_loader)}], Time: {(et // 60):.0f}m{(et % 60):.02f}s, Loss: {loss.item():.3f}')

                if (i + 1) % 10 == 0:
                    samples = self._generate(label, n_samples=label.shape[0], z_stds=[1.0])

                    images = make_grid(samples.cpu(), nrow=4, pad_value=1, normalize=True)
                    save_image(images, f'images/generated_sample_{step}.png')

                    self.save(f'weights/nf/{epoch + 1}.pth')

            if valid_loader:
                self.test(valid_loader)

    def test(self, test_loader: DataLoader) -> None:
        self.net.eval()

        with torch.no_grad():
            accuracy = 0.0

            for _, label in enumerate(test_loader):
                label = label.to(self.device)

                samples = self._generate(label, n_samples=label.shape[0], z_stds=[1.0])

                temp_accuracy = self.evaluator.eval(samples, label)
                accuracy += temp_accuracy

                print(f'\raccuracy: {temp_accuracy:.3f}', end='')

            accuracy /= len(test_loader)
            print(f'\raccuracy: {accuracy:.3f}')

    def save(self, net_name: str) -> None:
        torch.save(self.net, net_name)

    def load(self, net_name: str) -> None:
        if net_name:
            self.net = torch.load(net_name)
            self.optimizer.param_groups[0]['params'] = self.net.parameters()
