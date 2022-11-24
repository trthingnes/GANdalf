import datetime
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from gan import Discriminator, Generator
from dataset import MNIST
from util import get_device, save_state


class GAN:
    def __init__(
        self,
        n_epochs=30,
        lr=1e-4,
        batch_size=32,
        allow_cuda=True,
        seed=42,
        dataset=MNIST(),
    ):
        # Parameters
        self.n_epochs = n_epochs
        self.lr_g = self.lr_d = lr
        self.batch_size = batch_size
        self.device = get_device(allow_cuda, seed)

        self.noise_size = 10

        # Dataset and dataloader
        self.dataset = dataset
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        # Generator and discriminator models
        self.generator = Generator(
            img_size_in=self.noise_size,
            img_size_out=self.dataset.img_size,
        ).to(self.device)

        self.discriminator = Discriminator(
            img_size_in=self.dataset.img_size
        ).to(self.device)

        # Loss function and optimizers
        self.loss = nn.BCELoss()
        self.optim_g = Adam(self.generator.parameters(), lr=self.lr_g)
        self.optim_d = Adam(self.discriminator.parameters(), lr=self.lr_d)

        # Definitions of real [1, 1, 1, ...] and fake [0, 0, 0, ...] scores
        self.all_real_score = Variable(torch.ones(self.batch_size)).to(self.device)
        self.all_fake_score = Variable(torch.zeros(self.batch_size)).to(self.device)

    def generate_noise(self):
        """Generates a batch of 2D noise to be used in image generation."""
        return Variable(
            torch.randn(self.batch_size, self.noise_size, self.noise_size)
        ).to(self.device)

    def score_generated_images(self):
        """Generates a batch of images and gets them scored by the discriminator."""
        noise = self.generate_noise()

        # Generate random images
        images = self.generator(noise)

        # Return the discriminators score of these images
        return self.discriminator(images)

    def step_generator(self):
        """Updates the generator based on how well generated images are scored by the discriminator."""
        self.optim_g.zero_grad()

        # Check how the discriminator rates generated images
        # Compare to all being scored as real because this is the goal of the generator
        score = self.score_generated_images()
        loss = self.loss(score, self.all_real_score)

        loss.backward()
        self.optim_g.step()

        return loss.data.item()

    def step_discriminator(self, real_images):
        """Updates the discriminator based on how accurately it identifies real vs. fake images."""
        self.optim_d.zero_grad()

        # Check how the discriminator rates real images
        # Compare to all being scored as real since this is the goal of the discriminator
        real_loss = self.loss(
            self.discriminator(real_images), self.all_real_score
        )

        # Check how the discriminator rates generated images
        # Compare to all being scored as fake since this is the goal of the discriminator
        fake_loss = self.loss(self.score_generated_images(), self.all_fake_score)

        # Total loss reflects how many total images the discriminator gets wrong (false positive + negative)
        loss_d = real_loss + fake_loss

        # Use the loss to improve the model
        loss_d.backward()
        self.optim_d.step()

        return loss_d.data.item()

    def train(self):
        loss_g = loss_d = 0
        for epoch in range(1, self.n_epochs + 1):
            for (real_images, _) in self.dataloader:
                real_images = Variable(real_images).to(self.device)

                self.generator.train()  # Sets the generator into training mode.
                loss_g = self.step_generator()
                loss_d = self.step_discriminator(real_images)
                self.generator.eval()  # Sets the generator into evaluation mode.

            print(
                f"Epoch {epoch} -> Generator loss: {loss_g}, Discriminator loss: {loss_d}"
            )

            if epoch % (self.n_epochs // 10) == 0:
                self.save_models()

        self.save_models()

    def save_models(self):
        print("Saving models...")
        timestamp = str(datetime.datetime.utcnow()).replace(" ", "-")
        save_state(self.generator, f"generator_{timestamp}")
        save_state(self.discriminator, f"discriminator_{timestamp}")

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s: %(message)s",
    encoding="utf-8",
    level=logging.DEBUG,
    handlers=[logging.FileHandler("gandalf.log"), logging.StreamHandler(sys.stdout)],
)

GAN().train()
