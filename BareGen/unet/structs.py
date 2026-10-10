from ..base_structs import *
from ..attention import *


class SkipUpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode="conv", resnet_num=2, attention=False, dropout_p=.1):
        super(SkipUpSampleBlock, self).__init__()
        assert mode in ["conv", "inter"], "只支持conv和inter两种采样方式"
        if not attention:
            self.self_attn = nn.Identity()
        else:
            self.self_attn = ImageSelfAttentionBlockWithRotary(channels=out_channels * 2, max_freq=10,
                                                               query_key_norm_type="layer_norm", dropout_p=dropout_p)
        if mode == "conv":
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2,
                                         padding=1)
        else:
            self.up = Interpolate(in_channels=in_channels, out_channels=out_channels, mode="nearest", scale_factor=2)
        resnet = []
        for i in range(resnet_num):
            if i == 0:
                in_c = out_channels * 2
                out_c = out_channels
            else:
                in_c = out_channels
                out_c = out_channels
            resnet.append(ResidualBlock(in_channels=in_c, out_channels=out_c))
        self.resnet = nn.Sequential(*resnet)

    def skip_fusion(self, x, y):
        return torch.cat([x, y], 1)

    def forward(self, x, y):
        x = self.up(x)
        x = self.skip_fusion(x, y)
        x = x + self.self_attn(x)
        x = self.resnet(x)
        return x


class SkipDecoder(nn.Module):
    def __init__(self, attentions, channels, mode, resnet_num, dropout_p):
        super(SkipDecoder, self).__init__()
        if len(attentions) == 0:
            attentions = [False] * len(channels)
        else:
            if len(attentions) < len(channels):
                attentions = attentions + [False] * (len(channels) - len(attentions))
            else:
                attentions = attentions[:len(channels)]
        self.layer = nn.ModuleList()
        for cnt, item in enumerate(channels):
            self.layer.append(
                SkipUpSampleBlock(in_channels=item[0], out_channels=item[1], mode=mode, resnet_num=resnet_num,
                                  attention=attentions[cnt], dropout_p=dropout_p))

    def forward(self, x, encoder_feature):
        for cnt, layer in enumerate(self.layer):
            feature = encoder_feature.pop()
            x = layer(x=x, y=feature)
        return x
