import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_size_in):
        super().__init__()
        self.img_size_in = img_size_in
        self.img_pixels_in = img_size_in**2

        # A nice rule could be the second number is half the first one.
        self.embedding_size = 10
        self.negative_slope = 0.2

        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, stride=(2, 2), kernel_size=(2, 2)),
            nn.LeakyReLU(self.negative_slope, inplace=True),
        )   

        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, stride=(2, 2), kernel_size=(2, 2)),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # Input 7x7, output 3x3
        )

        # Collapse channels into single image.
        self.output_layer = nn.Sequential(
            nn.Flatten(),  # Flatten channels
            nn.Dropout(p=0.2),
            nn.Linear(1152, 1),
            nn.Sigmoid(),
        )

    def forward(self, images, labels):
        """Takes a list of images with labels and returns the probability of them being real."""
        images = self.hidden_layer1(images)  # Input: 28x28, output: 14x14
        images = self.hidden_layer2(images)  # Input: 14x14, output 7x7
        output = self.output_layer(images).squeeze()

        return output
