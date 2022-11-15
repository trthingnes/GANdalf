import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_size_in, n_labels, kernel_size):
        super().__init__()
        self.size = img_size_in
        self.n_pixels = img_size_in**2
        self.n_labels = n_labels
        self.kernel_size = kernel_size

        # A nice rule could be the second number is half the first one.
        # Question: Can this embedding layer be different from the generators?
        self.embedding_size = int(0.5 * self.n_labels)
        self.label_embedding = nn.Embedding(self.n_labels, self.embedding_size)

        self.model = nn.Sequential(
            # f(*(a, b, c)) = f(a, b, c)
            *self.hidden_layer(
                self.n_pixels + self.embedding_size, 1024, self.kernel_size
            ),
            *self.hidden_layer(1024, 512, self.kernel_size),
            *self.hidden_layer(512, 256, self.kernel_size),
            nn.Conv2d(256, 1, self.kernel_size),
            nn.Sigmoid()
        )

    @staticmethod
    def hidden_layer(in_features, out_features, kernel_size):
        negative_slope, dropout = 0.2, 0.3
        return (
            nn.Conv2d(
                in_features, out_features, kernel_size
            ),  # 2D Convolutional layer: (see lecture 3: CNN)
            nn.LeakyReLU(
                negative_slope, inplace=True
            ),  # Negative values get shrinked, see docs.
            nn.Dropout(
                dropout
            ),  # Dropout is useful for avoiding overfitting (see lecture 3: CNN)
        )

    def forward(self, images, labels):
        """
        Takes a list of images with labels and returns the probability of them being real.
        :param images: Tensor representation of n images. Shape: (n, sqrt(n_pixels), sqrt(n_pixels))
        :param labels: Labels belonging to the images. Shape: (n)
        :return: Probability of each image being real. Shape: (n)
        """
        n_images = images.size(
            0
        )  # Images is 3 dimentions, the first of which is the number of images
        images = images.view(
            n_images, self.n_pixels
        )  # Turn the 2D images into a 1D list of numbers
        label_dummies = self.label_embedding(
            labels
        )  # A number -> List of numbers. Shape: (n_labels)

        # Add the label dummy variables to the images array before inputting into the model.
        images = torch.cat([images, label_dummies], 1)

        return self.model(
            images
        ).squeeze()  # Squeeze remove all single element dimensions.
