import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, g_hidden, image_channel):
        super(Generator, self).__init__()
        self.kernel_size = 4
        self.main = nn.Sequential(
            # Input layer
            nn.ConvTranspose2d(z_dim, g_hidden * 8, self.kernel_size, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(g_hidden * 8),
            nn.ReLU(True),
            # 1st hidden layer
            *self.hidden_layer(g_hidden * 8, g_hidden * 4),
            # 2nd hidden layer
            *self.hidden_layer(g_hidden * 4, g_hidden * 2),
            # 3rd hidden layer
            self.hidden_layer(g_hidden * 2, g_hidden),
            # Output layer
            nn.ConvTranspose2d(g_hidden, image_channel, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def hidden_layer(self, in_features, out_features):
        stride = 2
        padding = 1
        return(
            nn.Conv2d(in_features, out_features, self.kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(True)
        )
    def forward(self, input_data):
        return self.main(input_data)