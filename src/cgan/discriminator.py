import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_pixels=784, n_labels=10):
        super().__init__()
        self.n_pixels = n_pixels
        self.n_labels = n_labels

        # A nice rule could be the second number is half the first one.
        self.label_embedding = nn.Embedding(n_labels, n_labels)

        self.model = nn.Sequential(
            *self.hidden_layer(n_pixels + n_labels, 1024),  # f(*(a, b, c)) = f(a, b, c)
            *self.hidden_layer(1024, 512),
            *self.hidden_layer(512, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    @staticmethod
    def hidden_layer(in_features, out_features):
        negative_slope, dropout = 0.2, 0.3
        return (
            nn.Linear(
                in_features, out_features
            ),  # Linear model xW + b (see lecture 1: linear regression)
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
