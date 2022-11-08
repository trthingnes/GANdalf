import math
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, n_pixels_in=100, n_pixels_out=784, n_labels=10):
        super().__init__()
        self.n_pixels_in = n_pixels_in
        self.n_pixels_out = n_pixels_out
        self.n_labels = n_labels

        self.label_embedding = nn.Embedding(n_labels, n_labels)

        self.model = nn.Sequential(
            *self.hidden_layer(100 + n_labels, 256),
            *self.hidden_layer(256, 512),
            *self.hidden_layer(512, 1024),
            nn.Linear(1024, n_pixels_out),
            nn.Tanh()
        )

    @staticmethod
    def hidden_layer(in_features: int, out_features: int):
        negative_slope = 0.2
        return (
            nn.Linear(in_features, out_features),  # Linear model xW + b (see lecture 1: linear regression)
            nn.LeakyReLU(negative_slope, inplace=True),  # Negative values get shrinked, see docs.
        )

    def forward(self, noise, labels):
        """
        Takes a list of noise images with labels and returns generated images.
        :param noise: Noise vectors to build images from. Shape: (n, sqrt(n_pixels_in), sqrt(n_pixels_in))
        :param labels: Labels belonging to the images. Shape: (n)
        :return: Generated images. Shape: (n, sqrt(n_pixels_out), sqrt(n_pixels_out))
        """
        n_images = noise.size(0)  # Noise is 3 dimentions, the first of which is the number of noise images
        noise = noise.view(n_images, self.n_pixels_in)  # Turn the 2D noise "images" into a 1D list of numbers
        label_dummies = self.label_embedding(labels)  # Get the "dummy" values for the label value

        images = torch.cat([noise, label_dummies], 1)  # Add the label data to each image
        size = int(math.sqrt(self.n_pixels_out))  # The width and height of the image is the square root of n_pixels_out
        return self.model(images).view(images.size(0), size, size)  # Return a 3D list of images.
