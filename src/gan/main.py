# Example GAN for generating MNIST images
# Article: https://medium.com/@simple.schwarz/b9bf71269da8
# Source code: https://github.com/SimpleSchwarz/GAN/blob/main/DCGAN_MNIST/DCGAN_MNIST.ipynb
import os
import sys
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from dataset import get_preprocessed_dataset
from models.generator import Generator
from models.discriminator import Discriminator
# Add project to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import get_device, save_state


class GAN:
    def __init__( 
        self,
        ALLOW_CUDA = True,
        DATA_PATH = "./training_data",
        BATCH_SIZE = 128, #Max recomended batch size apparently
        IMAGE_CHANNEL = 1, # grey scale
        Z_DIM = 100, # size of generator input
        G_HIDDEN = 64,
        X_DIM = 64,
        D_HIDDEN = 64,
        n_epochs = 1,
        REAL_LABEL = 1,
        FAKE_LABEL = 0,
        lr = 2e-4,
        seed = 1
    ):
        self.n_epochs = n_epochs
        self.device = get_device(ALLOW_CUDA, seed)
        dataset = get_preprocessed_dataset(DATA_PATH, X_DIM)
        self.dataloader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=2)

        #Generator
        self.generator = Generator(Z_DIM, G_HIDDEN, IMAGE_CHANNEL).to(self.device)
        self.generator.apply(self.weights_init)

        #Discriminator
        self.discriminator = Discriminator(IMAGE_CHANNEL, D_HIDDEN).to(self.device)
        self.discriminator.apply(self.weights_init)

        self.BCE_loss = nn.BCELoss()

        self.viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=self.device)

        #Optimizers for Generator and Discriminator
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.99))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.99))

        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.iters = 0

        self.REAL_LABEL = REAL_LABEL
        self.FAKE_LABEL = FAKE_LABEL
        self.Z_DIM = Z_DIM



    def weights_init(self, m):
        """Initializes weights based on classname of m."""
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        

    def trainingLoop(self):
        for epoch in range(self.n_epochs):
            for i, data in enumerate(self.dataloader, 0):

                # (1) Update the discriminator with real data
                self.discriminator.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.REAL_LABEL, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.BCE_loss(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_real_output = output.mean().item()

                # (2) Update the discriminator with fake data
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.Z_DIM, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.generator(noise)
                label.fill_(self.FAKE_LABEL)
                # Classify all fake batch with D
                output = self.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                disc_error_fake = self.BCE_loss(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                disc_error_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + disc_error_fake
                # Update D
                self.discriminator_optimizer.step()

                # (3) Update the generator with fake data
                self.generator.zero_grad()
                label.fill_(self.REAL_LABEL)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.BCE_loss(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.generator_optimizer.step()


                # Output training stats
                if i % 50 == 0:
                    print(
                        "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                        % (
                            epoch,
                            self.n_epochs,
                            i,
                            len(self.dataloader),
                            errD.item(),
                            errG.item(),
                            D_real_output,
                            D_G_z1,
                            D_G_z2,
                        )
                    )

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (self.iters % 500 == 0) or (
                    (epoch == self.n_epochs - 1) and (i == len(self.dataloader) - 1)
                ):
                    with torch.no_grad():
                        fake = self.generator(self.viz_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                self.iters += 1
        self.printing()
            


    def printing(self):
         # Grab a batch of real images from the dataloader
        real_batch = next(iter(self.dataloader)) 
            # Plot the real images
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(
                    real_batch[0].to(self.device)[:64], padding=5, normalize=True
                ).cpu(),
                (1, 2, 0),
            )
        )

        # Plot the fake images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(self.img_list[-1], (1, 2, 0)))
        plt.show()

GAN().trainingLoop()