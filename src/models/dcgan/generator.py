import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, img_size_in, img_size_out):
        super().__init__()
        self.img_size_in = img_size_in
        self.img_size_out = img_size_out
        self.img_pixels_in = img_size_in**2
        self.img_pixels_out = img_size_out**2
        self.embedding_size = 10

        self.noise_layer = nn.Sequential(
            nn.Linear(self.img_pixels_in, self.img_pixels_out), nn.ReLU(inplace=True)
        )

        self.hidden_layer1 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, stride=(2, 2), kernel_size=(4, 4)),
            nn.ReLU(inplace=True),
        )

        self.hidden_layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, stride=(2, 2), kernel_size=(4, 4)),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=(7, 7)), nn.Sigmoid()
        )

    def apply_noise_layer(self, n_images, noise):
        """Reshape the noise into several channels."""
        # Reshape the noise to be single strip of pixels
        noise = noise.view(n_images, self.img_pixels_in)

        # Apply the layer to the data
        noise = self.noise_layer(noise)

        return noise.view(
            n_images, 16, int(self.img_size_out / 4), int(self.img_size_out / 4)
        )

    def forward(self, noise):
        n_images = noise.size(0)

        # Apply initial dense (linear) layers
        noise = self.apply_noise_layer(n_images, noise)
        images = noise
        # Apply hidden layers
        images = self.hidden_layer1(images)  # Input: 7x7, output: 14x14
        images = self.hidden_layer2(images)  # Input: 14x14, output: 28x28

        # Collapse channels into single channel images
        return self.output_layer(images)
