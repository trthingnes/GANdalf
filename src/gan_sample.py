import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.datasets import MNIST

from gan import Generator
from util import load_state

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    required=True,
    help="The name of the generator model file to use (format: [name].pt)",
)
opt = parser.parse_args()

generator = load_state(
    Generator(img_size_in=10, img_size_out=28), opt.file
)
sqrt_samples = 5

noise = torch.randn(sqrt_samples**2, 100)
images_g = generator(noise).squeeze().detach().numpy()

fig = plt.figure(figsize=(sqrt_samples, sqrt_samples))
grid = ImageGrid(fig, 111, nrows_ncols=(sqrt_samples, sqrt_samples), axes_pad=0.3)

for ax, image in zip(grid,images_g):
    label_names = MNIST(root="training_data").classes
    ax.imshow(image, cmap="gray")

plt.show()
