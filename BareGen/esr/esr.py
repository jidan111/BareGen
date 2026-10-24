from .losses import *
from .structs import *


class RRDBNet(ConfigModule):
    def __init__(self, in_channels=3, hidden_channels=32, up_num=1, grow_channels=64,
                 layer_num=23, out_channels=None, up_mode="inter", loss_func=None, checkpoint_enable=False):
        super(RRDBNet, self).__init__()
        self.checkpoint_enable = checkpoint_enable
        if loss_func is None:
            self.loss_func = ESRLoss(using_perception=False, perception_weight=1., perception_net="alex",
                                     device=None)
        else:
            self.loss_func = loss_func
        if out_channels is None:
            out_channels = in_channels
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.rrd_layer = nn.Sequential(
            *[RRDB(in_channels=hidden_channels, hidden_channels=grow_channels) for i in range(layer_num)]
        )
        self.act = nn.LeakyReLU(.2)
        self.fusion = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1,
                                padding=1)
        up = []
        for i in range(up_num):
            up.append(RRDB(in_channels=hidden_channels, hidden_channels=grow_channels))
            up.append(UpSampleBlock(in_channels=hidden_channels, out_channels=hidden_channels, up_mode=up_mode))
            up.append(nn.LeakyReLU(.2))
        self.up = nn.Sequential(*up)
        self.hr_feature = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1,
                                    padding=1)
        self.out_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                  padding=1)

    def get_last_layer_weight(self):
        return self.out_conv.weight

    def __using_checkpoint(self, x, model):
        x = model(x)
        return x

    def forward(self, x):
        x = self.init_conv(x)
        if self.checkpoint_enable:
            h_ = checkpoint(
                self.__using_checkpoint,
                x, self.rrd_layer,
                use_reentrant=False
            )
        else:
            h_ = self.rrd_layer(x)
        h_ = self.act(self.fusion(h_))
        h_ = self.act(x + h_)
        if self.checkpoint_enable:
            h_ = checkpoint(
                self.__using_checkpoint,
                h_, self.up,
                use_reentrant=False
            )
        else:
            h_ = self.up(h_)
        h_ = self.act(self.hr_feature(h_))
        h_ = self.out_conv(h_)
        return h_

    def loss(self, x, target, return_features=False):
        inputs = self(x)
        loss = self.loss_func(inputs, target)
        if return_features:
            return loss, inputs
        return loss
