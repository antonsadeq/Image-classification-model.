import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, use_batch_normalization: bool, num_classes, kernel_size,
                 activation_function=nn.ReLU(), use_dropout=False):
        super(MyCNN, self).__init__()

        layers = []
        self.use_dropout = use_dropout
        for out_channels, k_size in zip(hidden_channels, kernel_size):
            layers.append(nn.Conv2d(input_channels, out_channels, kernel_size=k_size, padding=k_size // 2, bias=False))
            if use_batch_normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_function)
            if use_dropout:
                layers.append(nn.Dropout(0.1))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            input_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


num_classes = 20
model = MyCNN(
    input_channels=1,
    hidden_channels=[265, 512, 512, 1024],
    use_batch_normalization=True,
    num_classes=num_classes,
    kernel_size=[3, 3, 3, 3],
    activation_function=nn.ReLU(),
    use_dropout=True
)
