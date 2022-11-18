import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, img_size_in, img_size_out, n_labels):
        super().__init__()
        self.img_size_in = img_size_in
        self.img_size_out = img_size_out
        self.img_pixels_in = img_size_in**2
        self.img_pixels_out = img_size_out**2
        self.n_labels = n_labels
        self.embedding_size = 10
        self.negative_slope = 0.2

        print(self.img_size_out)

        self.noise_layer = nn.Sequential(
            # Dense layer xW + b (see lecture 1: linear regression)
            nn.Linear(self.img_pixels_in, self.img_pixels_out),
            # LeakyReLU is like ReLU but negative values get shrunk, not removed
            nn.LeakyReLU(self.negative_slope, inplace=True)
        )
        
        self.label_layer = nn.Sequential(
            # Embedding layer takes a label number 0-n and returns an array of numbers
            nn.Embedding(self.n_labels, self.embedding_size),
            nn.Linear(self.embedding_size, int(self.img_pixels_out/16)),
            nn.LeakyReLU(self.negative_slope, inplace=True)
        )
        
        self.hidden_layer1 = nn.Sequential(
            nn.ConvTranspose2d(17, 16, stride=(2, 2), kernel_size=(4, 4)),
            nn.LeakyReLU(self.negative_slope, inplace=True)
        )

        self.hidden_layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, stride=(2, 2), kernel_size=(4, 4)),
            nn.LeakyReLU(self.negative_slope, inplace=True)
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=(7, 7)),
            nn.Tanh()
        )

    def apply_noise_layer(self, n_images, noise):
        """Reshape the noise into several channels."""
        # Reshape the noise to be single strip of pixels
        noise = noise.view(n_images, self.img_pixels_in)

        # Apply the layer to the data
        noise = self.noise_layer(noise)

        return noise.view(n_images, int(self.img_pixels_out/(7*7)), 7, 7)

    def forward(self, noise, labels):
        """Takes a list of noise images with labels and returns generated images."""
        n_images = noise.size(0)

        # Apply initial dense (linear) layers
        noise = self.apply_noise_layer(n_images, noise)
        labels = self.label_layer(labels).view(n_images, 1, 7, 7)

        # Combine noise channels with label channel (16 noise channels + 1 label channel = 17 total)
        images = torch.cat([noise, labels], dim=1)
        
        # Apply hidden layers
        images = self.hidden_layer1(images)  # Input: 7x7, output: 14x14
        images = self.hidden_layer2(images)  # Input: 14x14, output: 28x28

        # Collapse channels into single channel images
        return self.output_layer(images)
