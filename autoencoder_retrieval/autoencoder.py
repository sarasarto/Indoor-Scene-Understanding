import torch.nn as nn


# Define network model (convolution autoencoder)
class EncodeModel(nn.Module):
    def __init__(self, judge=True):
        super(EncodeModel, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256*7*7

        )

        self.decode = nn.Sequential(

            nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, (2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 3, (2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, judge):
        enOutputs = self.encode(x)
        outputs = self.decode(enOutputs)
        if judge:
            return outputs
        else:
            return enOutputs


