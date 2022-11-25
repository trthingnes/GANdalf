import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.datasets import MNIST

from dcgan import Generator
from util import load_state

parser = argparse.ArgumentParser()
parser.add_argument(
    "--timestamp",
    required=True,
    help="The timestamp on the generator model to use (format: generator_[timestamp].pt)",
)
opt = parser.parse_args()

generator = load_state(
    nn.DataParallel(Generator(img_size_in=10, img_size_out=28)),
    f"generator_{opt.timestamp}",
)
sqrt_samples = 5

noise = torch.randn(sqrt_samples**2, 100)
images_g = generator(noise).squeeze().cpu().detach().numpy()

fig = plt.figure(figsize=(sqrt_samples, sqrt_samples))
grid = ImageGrid(fig, 111, nrows_ncols=(sqrt_samples, sqrt_samples), axes_pad=0.3)

for ax, image in zip(grid, images_g):
    ax.imshow(image, cmap="gray")

plt.show()
