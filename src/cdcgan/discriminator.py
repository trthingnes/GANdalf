import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_size_in, n_labels):
        super().__init__()
        self.img_size_in = img_size_in
        self.img_pixels_in = img_size_in**2
        self.n_labels = n_labels

        # A nice rule could be the second number is half the first one.
        self.embedding_size = 10
        self.negative_slope = 0.2

        # Make a label channel that can be combined with the noise.
        self.label_layer = nn.Sequential(
            # Embedding layer takes a label number 0-n and returns an array of numbers
            nn.Embedding(self.n_labels, self.embedding_size),
            nn.Linear(self.embedding_size, self.img_pixels_in),
        )

        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(2, 64, stride=(2, 2), kernel_size=(2, 2)),
            nn.LeakyReLU(self.negative_slope, inplace=True),
        )

        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, stride=(2, 2), kernel_size=(2, 2)),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # Input 7x7, output 3x3
        )

        # Collapse channels into single image.
        self.output_layer = nn.Sequential(
            nn.Flatten(),  # Flatten channels
            nn.Dropout(p=0.2),
            nn.Linear(1152, 1),
            nn.Sigmoid(),
        )

    def forward(self, images, labels):
        """Takes a list of images with labels and returns the probability of them being real."""
        n_images = images.size(0)

        labels = self.label_layer(labels).view(
            n_images, 1, self.img_size_in, self.img_size_in
        )

        # Combine noise channels with label channel (1 image channel + 1 label channel = 2 total)
        images = torch.cat([images, labels], dim=1)
        images = self.hidden_layer1(images)  # Input: 28x28, output: 14x14
        images = self.hidden_layer2(images)  # Input: 14x14, output 7x7
        output = self.output_layer(images).squeeze()

        return output
