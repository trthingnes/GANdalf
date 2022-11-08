import sys
from os import path

# Add project to path to allow imports
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import time
from util import get_device, save_model
from cgan.model import Generator, Discriminator


class CGAN:
    def __init__(self, n_epochs=5, lr=1e-4, seed=42, batch_size=32, noise_size=100, allow_cuda=True):
        self.n_epochs = n_epochs
        self.lr_g = self.lr_d = lr
        self.device = get_device(allow_cuda, seed)

        # Define dataset and dataloader
        dataset = datasets.FashionMNIST(root="training_data", transform=transforms.Compose([
            transforms.ToTensor(),
            # Question: Why do we do this?
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]), download=True)
        self.n_labels = len(dataset.classes)
        self.batch_size = batch_size
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define models
        self.noise_size = noise_size
        self.generator = Generator(n_pixels_in=noise_size).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Define optimizers and loss function
        # We have to use this loss function to get anything to work.
        self.loss = nn.BCELoss()
        self.optim_g = optim.Adam(self.generator.parameters(), lr=self.lr_g)
        self.optim_d = optim.Adam(
            self.discriminator.parameters(), lr=self.lr_d)

    def score_generated_images(self):
        """Generates a batch of images and gets them scored by the discriminator."""
        noise = Variable(torch.randn(self.batch_size, self.noise_size)).to(self.device)  # Question: Variable?
        labels_g = Variable(torch.LongTensor(np.random.randint(0, self.n_labels, self.batch_size))).to(self.device)
        images_g = self.generator(noise, labels_g)

        return self.discriminator(images_g, labels_g)

    def step_generator(self):
        """Updates the generator based on how well generated images are scored by the discriminator."""
        self.optim_g.zero_grad()

        score_g = self.score_generated_images()
        loss_g = self.loss(score_g, Variable(torch.ones(self.batch_size)).to(self.device))
        loss_g.backward()

        # Question: How does the optimizer get the loss score? Does it happen under the hood?
        self.optim_g.step()

        return loss_g.data.item()

    def step_discriminator(self, real_images, real_labels):
        """Updates the discriminator based on how accurately it identifies real vs. fake images."""
        self.optim_d.zero_grad()

        # Check how the discriminator rates real images
        real_score = self.discriminator(real_images, real_labels)
        real_loss = self.loss(real_score, Variable(
            torch.ones(self.batch_size)).to(self.device))  # ones = real

        # Check how the discriminator rates generated images
        fake_score = self.score_generated_images()
        fake_loss = self.loss(fake_score, Variable(torch.zeros(
            self.batch_size)).to(self.device))  # zeros = fake

        # Sum up the total loss
        loss_d = real_loss + fake_loss
        loss_d.backward()

        self.optim_d.step()

        return loss_d.data.item()

    def train(self):
        loss_g = loss_d = 0
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch}:")
            for (real_images, real_labels) in self.dataloader:
                real_images = Variable(real_images).to(self.device)
                real_labels = Variable(real_labels).to(self.device)

                # Sets the generator into training mode.
                self.generator.train()
                loss_g = self.step_generator()
                loss_d = self.step_discriminator(real_images, real_labels)
                self.generator.eval()

            print(f"Generator loss: {loss_g}, Discriminator loss: {loss_d}")

        print("Saving models...")
        timestamp = time.ctime().replace(" ", "-")
        save_model(self.generator, f"generator_{timestamp}")
        save_model(self.discriminator, f"discriminator_{timestamp}")


CGAN().train()