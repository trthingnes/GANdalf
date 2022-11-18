import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid

from cgan import Generator
from dataset import FashionMNIST
from util import load_state

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    required=True,
    help="The name of the generator model file to use (format: [name].pt)",
)
opt = parser.parse_args()

generator = load_state(Generator(), opt.file)


sqrt_samples = 5
noise = torch.randn(sqrt_samples**2, 100)
labels_g = torch.LongTensor(np.random.randint(0, 10, sqrt_samples**2))
images_g = generator(noise, labels_g).detach().numpy()

fig = plt.figure(figsize=(sqrt_samples, sqrt_samples))
grid = ImageGrid(fig, 111, nrows_ncols=(sqrt_samples, sqrt_samples), axes_pad=0.3)

for ax, label, image in zip(grid, labels_g, images_g):
    label_names = FashionMNIST().classes
    ax.set_title(label_names[label.item()])
    ax.imshow(image)

plt.show()
