# Example GAN for generating MNIST images
# Article: https://medium.com/@simple.schwarz/b9bf71269da8
# Source code: https://github.com/SimpleSchwarz/GAN/blob/main/DCGAN_MNIST/DCGAN_MNIST.ipynb

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from dataset import get_preprocessed_dataset
from models.generator import Generator
from models.discriminator import Discriminator
from util import get_device, weights_init


if __name__ == '__main__':

    ALLOW_CUDA = False
    DATA_PATH = "./training_data"
    BATCH_SIZE = 128
    IMAGE_CHANNEL = 1
    Z_DIM = 100
    G_HIDDEN = 64
    X_DIM = 64
    D_HIDDEN = 64
    EPOCH_NUM = 5
    REAL_LABEL = 1
    FAKE_LABEL = 0
    lr = 2e-4
    seed = 1

    # Setup device (using CUDA if allowed and available)
    device = get_device(ALLOW_CUDA, seed)

    # Preprocess data
    dataset = get_preprocessed_dataset(DATA_PATH, X_DIM)

    # Load data using a dataloader to not overflow the memory
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    # Create the generator
    netG = Generator(Z_DIM, G_HIDDEN, IMAGE_CHANNEL).to(device)
    netG.apply(weights_init)
    print(netG)

    # Create the discriminator
    netD = Discriminator(IMAGE_CHANNEL, D_HIDDEN).to(device)
    netD.apply(weights_init)
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that I will use to visualize the progression of the generator
    viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))


    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(EPOCH_NUM):
        for i, data in enumerate(dataloader, 0):

            # (1) Update the discriminator with real data
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # (2) Update the discriminator with fake data
            # Generate batch of latent vectors
            noise = torch.randn(b_size, Z_DIM, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(FAKE_LABEL)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # (3) Update the generator with fake data
            netG.zero_grad()
            label.fill_(REAL_LABEL)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        epoch,
                        EPOCH_NUM,
                        i,
                        len(dataloader),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or (
                (epoch == EPOCH_NUM - 1) and (i == len(dataloader) - 1)
            ):
                with torch.no_grad():
                    fake = netG(viz_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1


    # Validation

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64], padding=5, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()
