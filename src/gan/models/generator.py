import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, g_hidden, image_channel):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input layer
            nn.ConvTranspose2d(z_dim, g_hidden * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_hidden * 8),
            nn.ReLU(True),
            # 1st hidden layer
            nn.ConvTranspose2d(g_hidden * 8, g_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden * 4),
            nn.ReLU(True),
            # 2nd hidden layer
            nn.ConvTranspose2d(g_hidden * 4, g_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden * 2),
            nn.ReLU(True),
            # 3rd hidden layer
            nn.ConvTranspose2d(g_hidden * 2, g_hidden, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden),
            nn.ReLU(True),
            # Output layer
            nn.ConvTranspose2d(g_hidden, image_channel, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input_data):
        return self.main(input_data)