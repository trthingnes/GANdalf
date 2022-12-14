import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import ImageGrid

from models.cdcgan import Generator
from util import load_state

parser = argparse.ArgumentParser()
parser.add_argument(
    "--timestamp",
    required=True,
    help="The timestamp on the generator model to use (format: generator_[timestamp].pt)",
)
opt = parser.parse_args()

generator = load_state(
    nn.DataParallel(Generator(img_size_in=10, img_size_out=28, n_labels=10)),
    f"generator_{opt.timestamp}",
)

sqrt_samples = 10

noise = torch.randn(sqrt_samples**2, 100)
labels_g = torch.LongTensor([i % 10 for i in range(0, sqrt_samples**2)])
images_g = generator(noise, labels_g).squeeze().cpu().detach().numpy()

fig = plt.figure(figsize=(sqrt_samples, sqrt_samples))
grid = ImageGrid(fig, 111, nrows_ncols=(sqrt_samples, sqrt_samples), axes_pad=0.0)

for ax, label, image in zip(grid, labels_g, images_g):
    ax.imshow(image, cmap="gray")

plt.show()
