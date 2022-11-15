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
            # f(*(a, b, c)) is the same as f(a, b, c)
            # TODO: Insert a single linear layer first to account for the label embedding
            *self.hidden_layer(100 + self.img_pixels_in, 512, self.kernel_size),
            *self.hidden_layer(512, 1024, self.kernel_size),
            *self.hidden_layer(1024, 2048, self.kernel_size),
            nn.Conv2d(2048, self.img_pixels_out, self.kernel_size),
            nn.Tanh()
        )

    @staticmethod
    def first_layer(in_features, out_features):
        return (
            nn.Linear(in_features, out_features)
            # TODO: Make the first linear layer.
        )

    @staticmethod
    def hidden_layer(in_features, out_features, kernel_size):
        negative_slope = 0.2
        return (
            nn.Conv2d(
                in_features, out_features, kernel_size
            ),  # Convolutional layer: (see lecture 3: CNN)
            nn.LeakyReLU(
                negative_slope, inplace=True
            ),  # Negative values get shrinked, see docs.
        )

    def forward(self, noise, labels):
        """
        Takes a list of noise images with labels and returns generated images.
        :param noise: Noise vectors to build images from. Shape: (n, sqrt(n_pixels_in), sqrt(n_pixels_in))
        :param labels: Labels belonging to the images. Shape: (n)
        :return: Generated images. Shape: (n, sqrt(n_pixels_out), sqrt(n_pixels_out))
        """
        n_images = noise.size(
            0
        )  # The first dimension of the tensor is the number of images.

        noise = noise.view(
            n_images, self.img_size_in, self.img_size_in
        )  # Reshape the noise to be a 2D "picture"

        # This embedding has the same shape as the input noise. Question: What other options do we have?
        # Reshape the label embedding to be a 2D "picture"
        label_embedding = self.label_embedding(labels).view(
            n_images, self.img_size_in, self.img_size_in
        )

        images = torch.cat(
            [noise, label_embedding], 1
        )  # Add the label embedding to each image [Image|Embedding].

        return self.model(images).view(
            n_images, self.img_size_out, self.img_size_out
        )  # Return a 3D list of images.
