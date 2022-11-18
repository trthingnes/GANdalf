import matplotlib.pyplot as plt
import numpy as np
import torch
from model import Generator
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.datasets import FashionMNIST
from util import load_state

generator = load_state(Generator(img_size_in=10, img_size_out=28, n_labels=10), "generator_2022-11-18-13:49:49.654385")


sqrt_samples = 5
noise = torch.randn(sqrt_samples**2, 100)
labels_g = torch.LongTensor(np.random.randint(0, 10, sqrt_samples**2))
images_g = generator(noise, labels_g).squeeze().detach().numpy()

fig = plt.figure(figsize=(sqrt_samples, sqrt_samples))
grid = ImageGrid(fig, 111, nrows_ncols=(sqrt_samples, sqrt_samples), axes_pad=0.3)

for ax, label, image in zip(grid, labels_g, images_g):
    label_names = FashionMNIST(root="training_data").classes
    ax.set_title(label_names[label.item()])
    ax.imshow(image)

plt.show()
