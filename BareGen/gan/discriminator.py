from ..base_structs import *


class PatchDiscriminator(ConfigModule):
    def __init__(self, in_channels, channels, depth, checkpoint_enable=False):
        super(PatchDiscriminator, self).__init__()
        self.checkpoint_enable = checkpoint_enable
        channels = get_channels_array(channels, depth)[0]
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=channels[0][0], kernel_size=4, stride=1,
                                   padding=1)
        layer = []
        for cnt, item in enumerate(channels):
            layer.append(nn.Conv2d(in_channels=item[0], out_channels=item[1], kernel_size=4, stride=2, padding=1))
            layer.append(nn.BatchNorm2d(item[1]))
            layer.append(nn.LeakyReLU(0.2))
        self.layer = nn.ModuleList(layer)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=channels[-1][1], out_channels=channels[-1][1], kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(channels[-1][1]),
            nn.LeakyReLU(0.2),
        )
        self.out_conv = nn.Conv2d(in_channels=channels[-1][1], out_channels=1, kernel_size=4, stride=1, padding=1)

    def __using_checkpoint(self, x, model):
        x = model(x)
        return x

    def forward(self, x):
        x = self.init_conv(x)
        for layer in self.layer:
            if not isinstance(layer, nn.BatchNorm2d):
                if self.checkpoint_enable:
                    x = checkpoint(
                        self.__using_checkpoint,
                        x, layer,
                        use_reentrant=False
                    )
                else:
                    x = layer(x)
            else:
                x = layer(x)
        if self.checkpoint_enable:
            x = checkpoint(
                self.__using_checkpoint,
                x, self.fusion,
                use_reentrant=False
            )
        else:
            x = self.fusion(x)
        x = self.out_conv(x)
        return x

    def loss(self):
        pass


def get_depth(size, depth, min_size=4) -> int:
    for i in range(depth):
        if size % 2 == 0:
            size //= 2
        elif size <= min_size:
            break
        else:
            break
    return size


class Discriminator(ConfigModule):
    def __init__(self, image_shape, channels, depth):
        super(Discriminator, self).__init__()
        channels = get_channels_array(channels, depth)[0]
        in_c = image_shape[0]
        size = (get_depth(size=image_shape[1], depth=depth, min_size=4),
                get_depth(size=image_shape[2], depth=depth, min_size=4))
        self.init_conv = nn.Conv2d(in_channels=in_c, out_channels=channels[0][0], kernel_size=1)
        layer = []
        for cnt, item in enumerate(channels):
            layer.append(nn.Conv2d(in_channels=item[0], out_channels=item[1], kernel_size=4, stride=2, padding=1))
            layer.append(nn.BatchNorm2d(item[1]))
            layer.append(nn.LeakyReLU(0.2))
        self.layer = nn.Sequential(*layer)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=channels[-1][1], out_channels=channels[-1][1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[-1][1]),
            nn.LeakyReLU(0.2),
        )
        self.out = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_features=channels[-1][1] * math.prod(size), out_features=1),
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer(x)
        x = self.fusion(x)
        x = self.out(x)
        return x

    def loss(self):
        pass
