from ..base_structs import *


def modulate_unet(x, shift, scale):
    """
    x * (1 + scale) + shift
    shift:[B,N]
    scale:[B,N]
    """
    return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class AdaTimestepNorm(nn.Module):
    def __init__(self, dim, condition_dim):
        super(AdaTimestepNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, condition):
        x = self.norm(x)
        condition = self.mlp(condition)
        scale, shift = condition.chunk(2, dim=-1)
        return x * (1 + scale) + shift


class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels, condition_dim):
        super(AdaGroupNorm, self).__init__()
        self.norm = channels_get_norms(in_channels)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, in_channels * 2)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, condition):
        """
        x:[B,C,H,W]
        other:[B,N]
        """
        batch_size, *_ = condition.shape
        other = self.mlp(condition)
        scale, shift = other.chunk(2, dim=-1)
        x = self.norm(x)
        x = modulate_unet(x, scale, shift)
        return x


class ResidualConditionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim):
        super(ResidualConditionalBlock, self).__init__()
        self.norm1 = AdaGroupNorm(in_channels=in_channels, condition_dim=condition_dim)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = AdaGroupNorm(in_channels=out_channels, condition_dim=condition_dim)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.conv_equal = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.conv_equal = IdentityContinue()

    def forward(self, x, condition):
        h_ = self.norm1(x, condition)
        h_ = self.act1(h_)
        h_ = self.conv1(h_)
        h_ = self.norm2(h_, condition)
        h_ = self.act2(h_)
        h_ = self.conv2(h_)
        x = self.conv_equal(x)
        return x + h_


class UpSampleConditionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition1_dim, condition2_dim=None, mode="conv", resnet_num=2,
                 attention=False,
                 dropout_p=.1, max_freq=10):
        super(UpSampleConditionalBlock, self).__init__()
        assert mode in ["conv", "inter"], "只支持conv和inter两种采样方式"
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
            resnet.append(ResidualConditionalBlock(in_channels=in_c, out_channels=out_c,
                                                   condition_dim=condition1_dim))
        self.resnet = nn.ModuleList(resnet)
        if not attention:
            self.self_attn = IdentityContinue()
        else:
            self.self_attn = ImageSelfAttentionBlockWithRotary(channels=out_channels * 2, max_freq=max_freq,
                                                               query_key_norm_type="layer_norm", dropout_p=dropout_p)
        if condition2_dim is not None:
            self.cross_attn = ImageCrossAttentionBlock(channels=out_channels * 2, key_dim=condition2_dim,
                                                       dropout_p=dropout_p,
                                                       query_key_norm_type="none", max_freq=max_freq)
        else:
            self.cross_attn = IdentityContinue()

    def skip_fusion(self, x, y, z=None, z_weight=0.):
        if z is not None:
            y = y + z_weight * z
        return torch.cat([x, y], dim=1)

    def forward(self, x, y, condition1, condition2=None, z=None, z_weight=0.):
        x = self.up(x)
        x = self.skip_fusion(x=x, y=y, z=z, z_weight=z_weight)
        x = x + self.cross_attn(x, condition2, condition2)
        x = x + self.self_attn(x)
        for layer in self.resnet:
            x = layer(x, condition1)
        return x


class DownSampleConditionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition1_dim, condition2_dim=None, mode="conv", resnet_num=2,
                 attention=False, dropout_p=.1, max_freq=10):
        super(DownSampleConditionalBlock, self).__init__()
        assert mode in ["conv", "inter"], "只支持conv和inter两种采样方式"
        resnet = []
        if not attention:
            self.self_attn = IdentityContinue()
        else:
            self.self_attn = ImageSelfAttentionBlockWithRotary(channels=out_channels, max_freq=max_freq,
                                                               query_key_norm_type="layer_norm", dropout_p=dropout_p)
        for i in range(resnet_num):
            resnet.append(ResidualConditionalBlock(in_channels=out_channels, out_channels=out_channels,
                                                   condition_dim=condition1_dim))
        self.resnet = nn.ModuleList(resnet)
        if condition2_dim is not None:
            self.cross_attn = ImageCrossAttentionBlock(channels=out_channels, key_dim=condition2_dim, max_freq=max_freq)
        else:
            self.cross_attn = IdentityContinue()
        if mode == "conv":
            self.down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                  padding=1)
        else:
            self.down = Interpolate(in_channels=in_channels, out_channels=out_channels, mode="nearest",
                                    scale_factor=0.5)

    def forward(self, x, condition1, condition2=None):
        x = self.down(x)
        x = x + self.cross_attn(x, condition2, condition2)
        x = x + self.self_attn(x)
        for layer in self.resnet:
            x = layer(x, condition1)
        return x


class MiddleSampleConditionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition1_dim, condition2_dim=None, dropout_p=.1, max_freq=10):
        super(MiddleSampleConditionalBlock, self).__init__()
        self.res1 = ResidualConditionalBlock(in_channels=in_channels, out_channels=out_channels,
                                             condition_dim=condition1_dim)
        self.self_attn = ImageSelfAttentionBlockWithRotary(channels=out_channels, max_freq=max_freq,
                                                           dropout_p=dropout_p, query_key_norm_type="none")
        if condition2_dim is not None:
            self.cross_attn = ImageCrossAttentionBlock(channels=out_channels, key_dim=condition2_dim,
                                                       dropout_p=dropout_p, query_key_norm_type="none",
                                                       max_freq=max_freq)
        else:
            self.cross_attn = IdentityContinue()
        self.res2 = ResidualConditionalBlock(in_channels=out_channels, out_channels=out_channels,
                                             condition_dim=condition1_dim)

    def forward(self, x, condition1, condition2=None):
        x = self.res1(x, condition1)
        x = x + self.cross_attn(x, condition2, condition2)
        x = x + self.self_attn(x)
        x = self.res2(x, condition1)
        return x


class EncoderConditional(nn.Module):
    def __init__(self, channels: list, condition1_dim, condition2_dim, mode="inter", resnet_num=2, attentions=[],
                 max_freq=10,
                 dropout_p=.1):
        super(EncoderConditional, self).__init__()
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
                DownSampleConditionalBlock(in_channels=item[0], out_channels=item[1], condition1_dim=condition1_dim,
                                           condition2_dim=condition2_dim, mode=mode, resnet_num=resnet_num,
                                           attention=attentions[cnt], dropout_p=dropout_p, max_freq=max_freq))

    def forward(self, x, condition1, condition2=None, return_intermediates=False):
        feature = [x]
        for layer in self.layer:
            x = layer(x, condition1, condition2)
            if return_intermediates:
                feature.append(x)
        if return_intermediates:
            return feature
        return x


class DecoderConditional(nn.Module):
    def __init__(self, channels: list, condition1_dim, condition2_dim, mode="inter", resnet_num=2, attentions=[],
                 max_freq=10,
                 dropout_p=.1):
        super(DecoderConditional, self).__init__()
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
                UpSampleConditionalBlock(in_channels=item[0], out_channels=item[1], condition1_dim=condition1_dim,
                                         condition2_dim=condition2_dim, mode=mode, resnet_num=resnet_num,
                                         attention=attentions[cnt], dropout_p=dropout_p, max_freq=max_freq))

    def forward(self, x, encoder_feature, condition1, condition2=None, z=None, z_weight=0.):
        z_ = None
        for cnt, layer in enumerate(self.layer):
            feature = encoder_feature.pop()
            if z is not None:
                z_ = z.pop()
            x = layer(x=x, y=feature, condition1=condition1, condition2=condition2, z=z_, z_weight=z_weight)
        return x


class UnetConditional(ConfigModule):
    def __init__(self, in_channels, hidden_channels, timestep_dim, condition_dim=None, attentions=[], depth=3,
                 sample_mode="inter", resnet_num=2, max_freq=10,
                 dropout_p=.1):
        super(UnetConditional, self).__init__()
        encoder_channels, decoder_channels = get_channels_array(hidden_channels, layer_nums=depth)
        if condition_dim is not None:
            self.timestep_ln = AdaTimestepNorm(dim=timestep_dim, condition_dim=condition_dim)
        else:
            self.timestep_ln = IdentityContinue()
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=encoder_channels[0][0], kernel_size=1)

        self.encoder = EncoderConditional(channels=encoder_channels, condition1_dim=timestep_dim,
                                          condition2_dim=condition_dim, mode=sample_mode, resnet_num=resnet_num,
                                          attentions=attentions,
                                          max_freq=max_freq,
                                          dropout_p=dropout_p)
        self.decoder = DecoderConditional(channels=decoder_channels, condition1_dim=timestep_dim,
                                          condition2_dim=condition_dim, mode=sample_mode, resnet_num=resnet_num,
                                          attentions=attentions[::-1],
                                          max_freq=max_freq,
                                          dropout_p=dropout_p)
        self.middle_block = MiddleSampleConditionalBlock(in_channels=encoder_channels[-1][-1],
                                                         out_channels=decoder_channels[0][0],
                                                         condition1_dim=timestep_dim,
                                                         condition2_dim=condition_dim, dropout_p=dropout_p,
                                                         max_freq=max_freq)
        self.out_conv = nn.Conv2d(in_channels=decoder_channels[-1][-1], out_channels=in_channels, kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x, timestep, condition1=None, condition2=None, z=None, z_weight=0., **kwargs):
        """
        x:[B,C,H,W]
        timestep:[B,N]
        condition1:[B,S,D] CLIP的自然编码输出
        condition2:[B,D] CLIP的汇聚输出
        z:list[controlnet各层输出]
        z_weight:float
        """
        timestep = self.timestep_ln(timestep, condition2)
        x = self.init_conv(x)
        encode_feature = self.encoder(x, condition1=timestep, condition2=condition1, return_intermediates=True)
        up_feature = self.middle_block(encode_feature.pop(), condition1=timestep, condition2=condition1)
        up_feature = self.decoder(x=up_feature, encoder_feature=encode_feature, condition1=timestep,
                                  condition2=condition1,
                                  z=z, z_weight=z_weight)
        out = self.out_conv(up_feature)
        return out

    def get_last_layer_weight(self):
        return self.out_conv.weight

    def loss(self):
        pass
