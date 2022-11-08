import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_pixels, d_hidden):
        super(Discriminator, self).__init__()
        self.kernel_size = 4
        self.main = nn.Sequential(
            # Input layer
            nn.Conv2d(n_pixels, d_hidden, self.kernel_size, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
           
            # 1st Layer
            *self.hidden_layer(d_hidden, d_hidden * 2),
            # 2nd Layer
            *self.hidden_layer(d_hidden * 2, d_hidden * 4),
            # 3rd Layer
            *self.hidden_layer(d_hidden * 4, d_hidden * 8),

            # Output layer
            nn.Conv2d(d_hidden * 8, 1, self.kernel_size, bias=False),
            nn.Sigmoid(),
        )
    
    def hidden_layer(self, in_features, out_features):
        stride = 2
        padding = 1
        negative_slope = 0.2
        return(
            nn.Conv2d(in_features, out_features, self.kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(negative_slope, inplace=True)
        )
    
    def forward(self, input_data):
        return self.main(input_data).view(-1, 1).squeeze(1)

