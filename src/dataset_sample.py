import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import DataLoader

from dataset import MNIST

dataset = MNIST()
dataloader = DataLoader(dataset)
iterator = iter(dataloader)

# Get an image from each label
images = []
for i in range(10):
    image, label = next(iterator)
    while label.item() is not i:
        image, label = next(iterator)
    images.append(image.squeeze())

fig = plt.figure(figsize=(1, 10))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 10), axes_pad=0.0)
for ax, image in zip(grid, images):
    ax.imshow(image, cmap="gray")
plt.show()
