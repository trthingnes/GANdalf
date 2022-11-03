import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_pixels=784, n_labels=10):
        super().__init__()
        self.n_pixels = n_pixels
        self.n_labels = n_labels

        # From what I can gather, this is an overcomplicated way of making dummy variables.
        # The dataset of the code where we saw this has a number 0-9 as the first column.
        # What this embedding does is basically the same as making dummy variables [0,...,1]. Shape: (10).
        # The output from this seems random, but is the same if the input is the same.
        # I assume it therefore works since the model can consider these numbers instead of [0,...,1].
        # >>> le = nn.Embedding(3, 3)
        # >>> le(torch.tensor([0,0,1,1,2,2]))
        # tensor([[-0.9402, -1.5847, -0.3980], These are the same...
        #         [-0.9402, -1.5847, -0.3980],  ^
        #         [ 1.3810, -0.2916, -2.3109], these too...
        #         [ 1.3810, -0.2916, -2.3109],  ^
        #         [ 0.2044,  1.9303, -0.9796], and these
        #         [ 0.2044,  1.9303, -0.9796]]) ^
        # Question: Is this really the best way to do this?
        self.label_embedding = nn.Embedding(n_labels, n_labels)

        self.model = nn.Sequential(
            *self.hidden_layer(n_pixels + n_labels, 1024),
            *self.hidden_layer(1024, 512),
            *self.hidden_layer(512, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def hidden_layer(in_features: int, out_features: int):
        negative_slope, dropout = 0.2, 0.3
        return (
            nn.Linear(in_features, out_features),  # Linear model xW + b (see lecture 1: linear regression)
            nn.LeakyReLU(negative_slope, inplace=True),  # Negative values get shrinked, see docs.
            nn.Dropout(dropout)  # Dropout is useful for avoiding overfitting (see lecture 3: CNN)
        )

    def forward(self, images, labels):
        """
        Takes a list of images with labels and returns the probability of them being real.
        :param images: Tensor representation of n images. Shape: (n, sqrt(n_pixels), sqrt(n_pixels))
        :param labels: Labels belonging to the images. Shape: (n)
        :return: Probability of each image being real. Shape: (n)
        """
        n_images = images.size(0)  # Images is 3 dimentions, the first of which is the number of images
        images = images.view(n_images, self.n_pixels)  # Turn the 2D images into a 1D list of numbers
        label_dummies = self.label_embedding(labels)  # A number -> List of numbers. Shape: (n_labels)

        # Add the label dummy variables to the images array before inputting into the model.
        images = torch.cat([images, label_dummies], 1)

        return self.model(images).squeeze()  # Squeeze remove all single element dimensions.
