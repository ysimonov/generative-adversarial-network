import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels, num_features, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, num_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features*2, num_features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features*4, num_features*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, noise_channels, img_channels, num_features, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.ConvTranspose2d(noise_channels, num_features*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_features*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features*8, num_features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features*4, num_features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features*2, num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),        
            nn.ConvTranspose2d(num_features, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    generator = Generator(100, 3, 64, 1)
    discriminator = Discriminator(3, 64, 1)

    print(generator)
    print()
    print(discriminator)