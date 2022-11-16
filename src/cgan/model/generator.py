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

        self.embedding_size = int(0.5 * self.n_labels)
        self.label_embedding = nn.Embedding(self.n_labels, int(0.5 * self.n_labels))

        self.model = nn.Sequential(
            # f(*(a, b, c)) equals f(a, b, c)
            *self.first_layer(
                self.img_pixels_in + self.embedding_size, self.img_pixels_in
            ),
            *self.hidden_layer(self.img_pixels_in, 512, self.kernel_size),
            *self.hidden_layer(512, 1024, self.kernel_size),
            *self.hidden_layer(1024, 2048, self.kernel_size),
            *self.last_layer(2048, self.img_pixels_out, self.kernel_size),
        )

    @staticmethod
    def first_layer(in_features, out_features):
        negative_slope = 0.2
        return (
            # Dense layer xW + b (see lecture 1: linear regression)
            # TODO: Is this a sensible way to bake the label embedding into the data? Probably not.
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    @staticmethod
    def hidden_layer(in_features, out_features, kernel_size):
        negative_slope = 0.2
        return (
            # Convolutional layer (see lecture 3: CNN)
            # TODO: This is not in_features, it's in_channels. Very different.
            nn.Conv2d(in_features, out_features, kernel_size),
            # LeakyReLU is like ReLU but negative values get shrunk, not removed.
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    @staticmethod
    def last_layer(in_features, out_features, kernel_size):
        # TODO: In channels here too.
        return nn.Conv2d(in_features, out_features, kernel_size), nn.Tanh()

    def forward(self, noise, labels):
        """Takes a list of noise images with labels and returns generated images."""
        n_images = noise.size(0)

        # Reshape the noise to be single strip of pixels
        # TODO: We can't do this if we are going to send a 2D picture through the model.
        noise = noise.view(n_images, self.img_pixels_in)

        # Add the label embedding to each image [Image|Embedding]
        # TODO: If we are going to send 10*10 pixels into the conv layers, how will embedding work?
        images = torch.cat([noise, self.label_embedding(labels)], 1)

        # Reshape the strip of pixels into a 2D image
        return self.model(images).view(n_images, self.img_size_out, self.img_size_out)
