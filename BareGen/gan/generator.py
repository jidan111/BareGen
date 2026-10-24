from ..base_structs import *


class Generator(ConfigModule):
    def __init__(self, in_dim, base_size, channels, depth, out_channels, mode="inter",
                 resnet_num=2, attentions=[], dropout_p=0.1, checkpoint_enable=False):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        if type(base_size) == int:
            base_size = (base_size, base_size)
        channels = get_channels_array(channels, depth)[0]
        self.init_linear = nn.Sequential(
            nn.Linear(in_dim, channels[0][0] * math.prod(base_size)),
            nn.Unflatten(1, (channels[0][0], *base_size)),
        )
        self.decoder = Decoder(channels=channels, mode=mode, resnet_num=resnet_num,
                               attentions=attentions, dropout_p=dropout_p, checkpoint_enable=checkpoint_enable)
        self.out_conv = nn.Conv2d(in_channels=channels[-1][-1], out_channels=out_channels, kernel_size=3, stride=1,
                                  padding=1)

    def forward(self, x):
        x = self.init_linear(x)
        x = self.decoder(x)
        x = self.out_conv(x)
        return x

    def loss(self):
        pass

    @torch.no_grad()
    def sample(self, batch_size, device):
        noise = torch.randn(batch_size, self.in_dim).to(device)
        out = self(noise)
        return out
