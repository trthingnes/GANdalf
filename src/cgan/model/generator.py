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

        self.model = nn.Sequential(
            # f(*(a, b, c)) equals f(a, b, c)
            *self.noise_layer(
                self.img_pixels_in + self.embedding_size, self.img_pixels_in
            ),
            *self.hidden_layer(self.img_pixels_in, 512, self.kernel_size),
            *self.hidden_layer(512, 1024, self.kernel_size),
            *self.hidden_layer(1024, 2048, self.kernel_size),
            *self.last_layer(2048, self.img_pixels_out, self.kernel_size),
        )

        self.noise_layer = nn.Sequential(
            # Dense layer xW + b (see lecture 1: linear regression)
            nn.Linear(self.img_pixels_in, self.img_pixels_out/4),
            # LeakyReLU is like ReLU but negative values get shrunk, not removed
            nn.LeakyReLU(self.negative_slope, inplace=True)
        )

        self.label_layer = nn.Sequential(
            # Embedding layer takes a label number 0-n and returns an array of numbers
            nn.Embedding(self.n_labels, self.embedding_size),
            nn.Linear(self.embedding_size, self.img_pixels_out/4),
            nn.LeakyReLU(self.negative_slope, inplace=True)
        )

    def apply_noise_layer(self, n_images, noise):
        # Reshape the noise to be single strip of pixels
        noise = noise.view(n_images, self.img_pixels_in)

        # Apply the layer to the data
        noise = self.noise_layer(noise)

        return noise.view(n_images, self.img_pixels_out/4*7*7, 7, 7)
        
    
    def apply_label_layer(self, n_images, labels):
        return self.label_layer(labels).view()  # TODO

    @staticmethod
    def hidden_layer(in_features, out_features, kernel_size):
        return (
            # Convolutional layer (see lecture 3: CNN)
            # TODO: This is not in_features, it's in_channels. Very different.
            nn.Conv2d(in_features, out_features, kernel_size),
            # LeakyReLU is like ReLU but negative values get shrunk, not removed.
            nn.LeakyReLU(self.negative_slope, inplace=True),
        )

    @staticmethod
    def last_layer(in_features, out_features, kernel_size):
        # TODO: In channels here too.
        return nn.Conv2d(in_features, out_features, kernel_size), nn.Tanh()

    def forward(self, noise, labels):
        """Takes a list of noise images with labels and returns generated images."""
        n_images = noise.size(0)

        # Reshape the noise to be single strip of pixels
        noise = noise.view(n_images, self.img_pixels_in)

        noise = self.noise_layer(10*10, 7*7*10)(noise)

        # Add the label embedding to each image [Image|Embedding]
        # TODO: If we are going to send 10*10 pixels into the conv layers, how will embedding work?
        images = torch.cat([noise, self.label_embedding(labels)], 1)

        # Reshape the strip of pixels into a 2D image
        return self.model(images).view(n_images, self.img_size_out, self.img_size_out)
