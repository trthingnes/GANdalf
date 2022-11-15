import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_size_in, n_labels, kernel_size):
        super().__init__()
        self.img_size_in = img_size_in
        self.img_pixels_in = img_size_in**2
        self.n_labels = n_labels
        self.kernel_size = kernel_size

        # A nice rule could be the second number is half the first one.
        self.embedding_size = int(0.5 * self.n_labels)
        self.label_embedding = nn.Embedding(self.n_labels, self.embedding_size)

        self.model = nn.Sequential(
            # f(*(a, b, c)) equals f(a, b, c)
            *self.first_layer(
                self.img_pixels_in + self.embedding_size, self.img_pixels_in
            ),
            *self.hidden_layer(self.img_pixels_in, 1024, self.kernel_size),
            *self.hidden_layer(1024, 512, self.kernel_size),
            *self.hidden_layer(512, 256, self.kernel_size),
            *self.last_layer(256, 1, self.kernel_size),
        )

    @staticmethod
    def first_layer(in_features, out_features):
        negative_slope, dropout = 0.2, 0.3
        return (
            # Dense layer xW + b (see lecture 1: linear regression)
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Dropout(dropout),
        )

    @staticmethod
    def hidden_layer(in_features, out_features, kernel_size):
        negative_slope, dropout = 0.2, 0.3
        return (
            # 2D Convolutional layer (see lecture 3: CNN)
            nn.Conv2d(in_features, out_features, kernel_size),
            # LeakyReLU is like ReLU but negative values get shrunk, not removed.
            nn.LeakyReLU(negative_slope, inplace=True),
            # Dropout is useful for avoiding over-fitting (see lecture 3: CNN)
            nn.Dropout(dropout),
        )

    @staticmethod
    def last_layer(in_features, out_features, kernel_size):
        return nn.Conv2d(in_features, out_features, kernel_size), nn.Sigmoid()

    def forward(self, images, labels):
        """Takes a list of images with labels and returns the probability of them being real."""
        n_images = images.img_size_in(0)

        # Turn the 2D images into a 1D list of numbers
        images = images.view(n_images, self.img_pixels_in)

        # Add the label embedding to the images array before inputting into the model.
        images = torch.cat([images, self.label_embedding(labels)], 1)

        # Squeeze remove all single element dimensions.
        return self.model(images).squeeze()
