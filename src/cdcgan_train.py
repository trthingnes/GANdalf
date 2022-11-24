import argparse
import datetime
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from cdcgan import Discriminator, Generator
from dataset import FashionMNIST
from util import get_device, get_device_count, load_state, save_state


class CDCGAN:
    def __init__(self, continue_from_timestamp=None):
        # Parameters
        self.device = get_device(allow_cuda=True, seed=42)
        self.n_epochs = 100
        self.n_saves = 10
        self.lr_g = self.lr_d = 1e-4
        self.batch_size = 32 * get_device_count(cuda=(self.device.type == "cuda"))
        self.noise_size = 10

        # Dataset and dataloader
        self.dataset = FashionMNIST()
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        # Generator and discriminator models
        self.generator = nn.DataParallel(
            Generator(
                img_size_in=self.noise_size,
                img_size_out=self.dataset.img_size,
                n_labels=self.dataset.n_labels,
            )
        ).to(self.device)

        self.discriminator = nn.DataParallel(
            Discriminator(
                img_size_in=self.dataset.img_size,
                n_labels=self.dataset.n_labels,
            )
        ).to(self.device)

        # Load state if we are continuing training existing data
        if continue_from_timestamp:
            load_state(self.generator, f"generator_{continue_from_timestamp}")
            load_state(self.discriminator, f"discriminator_{continue_from_timestamp}")

        # Loss function and optimizers
        self.loss = nn.BCELoss()
        self.optim_g = Adam(self.generator.parameters(), lr=self.lr_g)
        self.optim_d = Adam(self.discriminator.parameters(), lr=self.lr_d)

        # Definitions of real [1, 1, 1, ...] and fake [0, 0, 0, ...] scores
        self.all_real_score = lambda size: Variable(torch.ones(size)).to(self.device)
        self.all_fake_score = lambda size: Variable(torch.zeros(size)).to(self.device)

    def generate_noise(self):
        """Generates a batch of 2D noise to be used in image generation."""
        return Variable(
            torch.randn(self.batch_size, self.noise_size, self.noise_size)
        ).to(self.device)

    def generate_labels(self):
        """Generates a batch of labels to be used in image generation."""
        return Variable(
            torch.LongTensor(
                np.random.randint(0, self.dataset.n_labels, self.batch_size)
            )
        ).to(self.device)

    def score_generated_images(self):
        """Generates a batch of images and gets them scored by the discriminator."""
        noise = self.generate_noise()
        labels = self.generate_labels()

        # Generate random images
        images = self.generator(noise, labels)

        # Return the discriminators score of these images
        return self.discriminator(images, labels)

    def step_generator(self):
        """Updates the generator based on how well generated images are scored by the discriminator."""
        self.optim_g.zero_grad()

        # Check how the discriminator rates generated images
        # Compare to all being scored as real because this is the goal of the generator
        score = self.score_generated_images()
        loss = self.loss(score, self.all_real_score(score.size(0)))

        loss.backward()
        self.optim_g.step()

        return loss.data.item()

    def step_discriminator(self, real_images, real_labels):
        """Updates the discriminator based on how accurately it identifies real vs. fake images."""
        self.optim_d.zero_grad()

        # Check how the discriminator rates real images
        # Compare to all being scored as real since this is the goal of the discriminator
        score = self.discriminator(real_images, real_labels)
        real_loss = self.loss(score, self.all_real_score(score.size(0)))

        # Check how the discriminator rates generated images
        # Compare to all being scored as fake since this is the goal of the discriminator
        score = self.score_generated_images()
        fake_loss = self.loss(score, self.all_fake_score(score.size(0)))

        # Total loss reflects how many total images the discriminator gets wrong (false positive + negative)
        loss_d = real_loss + fake_loss

        # Use the loss to improve the model
        loss_d.backward()
        self.optim_d.step()

        return loss_d.data.item()

    def train(self):
        loss_g = loss_d = 0
        for epoch in range(1, self.n_epochs + 1):
            for (real_images, real_labels) in self.dataloader:
                real_images = Variable(real_images).to(self.device)
                real_labels = Variable(real_labels).to(self.device)

                self.generator.train()  # Sets the generator into training mode.
                loss_g = self.step_generator()
                self.generator.eval()  # Sets the generator into evaluation mode.
                self.discriminator.train()
                loss_d = self.step_discriminator(real_images, real_labels)
                self.discriminator.eval()

            logging.info(
                f"Epoch {epoch} -> Generator loss: {loss_g}, Discriminator loss: {loss_d}"
            )

            if epoch % (self.n_epochs // self.n_saves) == 0 or epoch == self.n_epochs:
                self.save_models()

    def save_models(self):
        timestamp = str(datetime.datetime.utcnow()).replace(" ", "-")
        save_state(self.generator, f"generator_{timestamp}")
        save_state(self.discriminator, f"discriminator_{timestamp}")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--timestamp",
    required=False,
    help="The timestamp on the model to continue training (format: generator_[timestamp].pt)",
)
opt = parser.parse_args()

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s: %(message)s",
    encoding="utf-8",
    level=logging.DEBUG,
    handlers=[logging.FileHandler("gandalf.log"), logging.StreamHandler(sys.stdout)],
)

CDCGAN(continue_from_timestamp=opt.timestamp).train()
