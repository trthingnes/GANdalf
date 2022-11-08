import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import FashionMNIST
from model import Generator, Discriminator
from ..util import get_device, save_model


class CGAN:
    def __init__(self, n_epochs=30, lr=1e-4, seed=42, batch_size=32, noise_size=100, allow_cuda=True):
        self.n_epochs = n_epochs
        self.lr_g = self.lr_d = lr
        self.device = get_device(allow_cuda, seed)

        # Define dataset and dataloader
        dataset = FashionMNIST(transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))  # Question: Why do we do this?
        ]))
        self.batch_size = batch_size
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define models
        self.noise_size = noise_size
        self.generator = Generator(n_pixels_in=noise_size).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Define optimizers and loss function
        self.loss = nn.BCELoss()
        self.optim_g = optim.Adam(self.generator.parameters(), lr=self.lr_g)
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=self.lr_d)

    def score_generated_images(self):
        """Generates a batch of images and gets them scored by the discriminator."""
        noise = Variable(torch.randn(self.batch_size, self.noise_size)).to(self.device)  # Question: Variable?
        g_labels = Variable(torch.LongTensor(np.random.randint(0, 10, self.batch_size))).to(self.device)
        g_images = self.generator(noise, g_labels)

        return self.discriminator(g_images, g_labels)

    def step_generator(self):
        """Updates the generator based on how well generated images are scored by the discriminator."""
        self.optim_g.zero_grad()

        score_g = self.score_generated_images()
        loss_g = self.loss(score_g, Variable(torch.ones(self.batch_size)).to(self.device))
        loss_g.backward()

        self.optim_g.step()

        return loss_g.data[0]

    def step_discriminator(self, real_images, real_labels):
        """Updates the discriminator based on how accurately it identifies real vs. fake images."""
        self.optim_d.zero_grad()

        # Check how the discriminator rates real images
        real_score = self.discriminator(real_images, real_labels)
        real_loss = self.loss(real_score, Variable(torch.ones(self.batch_size)).to(self.device))  # ones = real

        # Check how the discriminator rates generated images
        fake_score = self.score_generated_images()
        fake_loss = self.loss(fake_score, Variable(torch.zeros(self.batch_size)).to(self.device))  # zeros = fake

        # Sum up the total loss
        loss_d = real_loss + fake_loss
        loss_d.backward()

        self.optim_d.step()

        return loss_d.data[0]

    def train(self):
        loss_g = loss_d = 0
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch}:")
            for i, (images, labels) in enumerate(self.dataloader):
                real_images = Variable(images).to(self.device)
                labels = Variable(labels).to(self.device)

                self.generator.train()
                loss_g = self.step_generator()
                loss_d = self.step_discriminator(real_images, labels)
                self.generator.eval()

            print(f"Generator loss: {loss_g}, Discriminator loss: {loss_d}")

        print("Saving models...")
        timestamp = time.ctime().replace(" ", "-")
        save_model(self.generator, f"generator_{timestamp}")
        save_model(self.discriminator, f"discriminator_{timestamp}")
