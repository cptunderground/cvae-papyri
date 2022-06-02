from torch import nn


class ConvAutoEncoder(nn.Module):
    """
    TODO: Class doc
    """
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        )

    def forward(self, image):
        encoding = self.encoder(image)
        decoding = self.decoder(encoding)

        return encoding, decoding
