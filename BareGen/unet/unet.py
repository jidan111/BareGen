from .structs import *


class Unet(ConfigModule):
    def __init__(self, in_channels, hidden_channels, attentions, depth, sample_mode="inter", resnet_num=2,
                 dropout_p=.1,out_channels=None, loss_func=None):
        super(Unet, self).__init__()
        if loss_func is None:
            self.loss_func = nn.MSELoss()
        else:
            self.loss_func = loss_func
        encoder_channels, decoder_channels = get_channels_array(hidden_channels, layer_nums=depth)
        out_channels = in_channels if out_channels is None else out_channels
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=encoder_channels[0][0], kernel_size=1)
        self.encoder = Encoder(channels=encoder_channels, mode=sample_mode, resnet_num=resnet_num,
                               attentions=attentions, dropout_p=dropout_p)
        self.decoder = SkipDecoder(channels=decoder_channels, mode=sample_mode,
                                   resnet_num=resnet_num,
                                   attentions=attentions[::-1], dropout_p=dropout_p)
        self.middle_block = nn.Sequential(
            ResidualBlock(in_channels=encoder_channels[-1][-1], out_channels=encoder_channels[-1][-1]),
            ImageSelfAttentionBlockWithRotary(encoder_channels[-1][-1], max_freq=10, dropout_p=.1,
                                              query_key_norm_type="none"),
            ResidualBlock(in_channels=encoder_channels[-1][-1], out_channels=encoder_channels[-1][-1]),
        )
        self.out_conv = nn.Conv2d(in_channels=decoder_channels[-1][-1], out_channels=out_channels, kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        x = self.init_conv(x)
        encode_feature = self.encoder(x, return_intermediates=True)
        up_feature = self.middle_block(encode_feature.pop())
        up_feature = self.decoder(up_feature, encode_feature)
        out = self.out_conv(up_feature)
        return out

    def loss(self, x):
        inputs = self(x)
        loss = self.loss_func(inputs, x)
        return loss

    def get_last_layer_weight(self):
        return self.out_conv.weight
