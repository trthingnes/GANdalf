import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, img_size_in, img_size_out, n_labels, kernel_size):
        super().__init__()
        self.img_size_in = img_size_in
        self.img_size_out = img_size_out
        self.img_pixels_in = img_size_in**2
        self.img_pixels_out = img_size_out**2
        self.n_labels = n_labels
        self.kernel_size = kernel_size
        
        self.negative_slope = 0.2
        self.embedding_size = 10

        self.label_embedding = nn.Embedding(self.n_labels, self.embedding_size)

        self.noise_layer = nn.Sequential(
            # Dense layer xW + b (see lecture 1: linear regression)
            nn.Linear(self.img_pixels_in, self.img_pixels_out),
            # LeakyReLU is like ReLU but negative values get shrunk, not removed
            nn.LeakyReLU(self.negative_slope, inplace=True)
        )

        self.label_layer = nn.Sequential(
            # Embedding layer takes a label number 0-n and returns an array of numbers
            nn.Embedding(self.n_labels, self.embedding_size),
            nn.Linear(self.embedding_size, self.img_pixels_out/16),
            nn.LeakyReLU(self.negative_slope, inplace=True)
        )

    def apply_noise_layer(self, n_images, noise):
        """Reshape and apply the noise layer to the noise data."""

        # Reshape the noise to be single strip of pixels
        noise = noise.view(n_images, self.img_pixels_in)

        # Apply the layer to the data
        noise = self.noise_layer(noise)

        return noise.view(n_images, self.img_pixels_out/(7*7), 7, 7)
        
    def apply_label_layer(self, n_images, labels):
        """Apply the label layer to the label data."""
        return self.label_layer(labels).view(n_images, 7, 7)

    def apply_hidden_layer(self, n_images, images, in_channels, out_channels):
        # Stride = (2, 2)
        # Kernel size = (4, 4)
        pass

    def apply_output_layer(self, n_images, images, in_channels):
        pass

    @staticmethod
    def last_layer(in_features, out_features, kernel_size):
        # TODO: In channels here too.
        return nn.Conv2d(in_features, out_features, kernel_size), nn.Tanh()

    def forward(self, noise, labels):
        """Takes a list of noise images with labels and returns generated images."""
        n_images = noise.size(0)

        # Apply initial dense (linear) layers
        noise = self.apply_noise_layer(n_images, noise)
        labels = self.apply_label_layer(n_images, labels)

        # Combine noise channels with label channel (16 noise channels + 1 label channels = 17 total)
        images = [torch.cat([noise[i], labels[i]]) for i in range(n_images)]   
        
        # Apply hidden layers
        images = self.apply_hidden_layer(n_images, images, 17, 16)
        images = self.apply_hidden_layer(n_images, images, 16, 16)

        # Collapse channels into single channel images
        return self.apply_output_layer(n_images, images, 16)
