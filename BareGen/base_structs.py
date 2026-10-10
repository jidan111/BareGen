from .import_packages import *
from .attention import *


def channels_get_norms(channels):
    per_num_channels = [8, 16, 32]
    for per_c in per_num_channels:
        if channels % per_c == 0:
            num_g = channels // per_c
            return nn.GroupNorm(num_groups=num_g, num_channels=channels)
    return nn.GroupNorm(num_groups=1, num_channels=channels)


def get_channels_array(channels, layer_nums) -> tuple:
    ins = []
    outs = []
    if type(channels) == int:
        ins = [channels * (2 ** i) for i in range(layer_nums)]
        outs = [channels * (2 ** i) for i in range(1, layer_nums + 1)]
    else:
        channels = list(channels)
        assert len(channels) == layer_nums + 1, \
            f"传入channels长度于layer_nums不匹配, channels={channels}, 对应层数为{len(channels) - 1}层, 但layer_nums={layer_nums}"
        ins = channels
        outs = channels[1:]
    encoder = list(zip(ins, outs))
    decoder = list(zip(outs, ins))[::-1]
    return encoder, decoder


class IdentityContinue(nn.Module):
    def __init__(self):
        super(IdentityContinue, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x


class ConfigModule(nn.Module):
    """
    自动捕获子类初始化参数，无需手动传递！
    所有子类只需要写：super().__init__() 即可
    加载子类只需要 Object(**config)即可加载子类，方便复现
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        subclass = self.__class__
        sig = inspect.signature(subclass.__init__)
        frame = inspect.currentframe().f_back  # 获取调用栈（子类 __init__）
        support_types = (str, int, float, list, dict, tuple, bool)
        local_vars = frame.f_locals
        params = list(sig.parameters.keys())[1:]
        config = {
            k: local_vars[k] for k in params if k in local_vars and isinstance(local_vars[k], support_types)
        }
        self.config = {self.__class__.__name__: config}

    def loss(self, *args, **kwargs):
        raise NotImplementedError


class EMA(object):
    def __init__(
            self,
            decay: float = 0.9999,
    ):
        self.decay = decay
        self.step = 0
        self.shadow = {}

    @torch.no_grad()
    def set_shadow(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        decay = 1 - (1 - self.decay) * (1 - 1 / max(self.step, 1))
        for name, param in model.named_parameters():
            self.shadow[name].mul_(decay).add_(
                param.data, alpha=1 - decay
            )

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module):
        backup = {}
        for name, param in model.named_parameters():
            backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup):
        for name, param in model.named_parameters():
            param.data.copy_(backup[name])

    def state_dict(self):
        return {
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
            "step": self.step,
        }

    def load_state_dict(self, state):
        self.shadow = state["shadow"]
        self.step = state["step"]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            channels_get_norms(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            channels_get_norms(out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )
        if in_channels != out_channels:
            self.conv_equal = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.conv_equal = IdentityContinue()

    def forward(self, x):
        h_ = self.conv1(x)
        x = self.conv_equal(x)
        return x + h_


class Interpolate(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2., mode='nearest'):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode="conv", resnet_num=2, max_freq=10, attention=False,
                 dropout_p=.1):
        super(UpSampleBlock, self).__init__()
        assert mode in ["conv", "inter"], "只支持conv和inter两种采样方式"
        if not attention:
            self.self_attn = IdentityContinue()
        else:
            self.self_attn = ImageSelfAttentionBlockWithRotary(channels=out_channels, max_freq=max_freq,
                                                               query_key_norm_type="layer_norm", dropout_p=dropout_p)
        if mode == "conv":
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2,
                                         padding=1)
        else:
            self.up = Interpolate(in_channels=in_channels, out_channels=out_channels, mode="nearest", scale_factor=2)
        resnet = []
        for i in range(resnet_num):
            resnet.append(ResidualBlock(in_channels=out_channels, out_channels=out_channels))
        self.resnet = nn.Sequential(*resnet)

    def forward(self, x):
        x = self.up(x)
        x = x + self.self_attn(x)
        x = self.resnet(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode="conv", resnet_num=2, max_freq=10, attention=False,
                 dropout_p=.1):
        super(DownSampleBlock, self).__init__()
        assert mode in ["conv", "inter"], "只支持conv和inter两种采样方式"
        resnet = []
        if not attention:
            self.self_attn = IdentityContinue()
        else:
            self.self_attn = ImageSelfAttentionBlockWithRotary(channels=out_channels, max_freq=max_freq,
                                                               query_key_norm_type="layer_norm", dropout_p=dropout_p)
        for i in range(resnet_num):
            resnet.append(ResidualBlock(in_channels=out_channels, out_channels=out_channels))
        self.resnet = nn.Sequential(*resnet)
        if mode == "conv":
            self.down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                  padding=1)
        else:
            self.down = Interpolate(in_channels=in_channels, out_channels=out_channels, mode="nearest",
                                    scale_factor=0.5)

    def forward(self, x):
        x = self.down(x)
        x = x + self.self_attn(x)
        x = self.resnet(x)
        return x


class Encoder(nn.Module):
    def __init__(self, channels: list, mode="inter", resnet_num=2, attentions=[], dropout_p=.1,
                 checkpoint_enable=False):
        super(Encoder, self).__init__()
        if len(attentions) == 0:
            attentions = [False] * len(channels)
        else:
            if len(attentions) < len(channels):
                attentions = attentions + [False] * (len(channels) - len(attentions))
            else:
                attentions = attentions[:len(channels)]
        self.checkpoint_enable = checkpoint_enable
        self.layer = nn.ModuleList()
        for cnt, item in enumerate(channels):
            self.layer.append(
                DownSampleBlock(in_channels=item[0], out_channels=item[1], mode=mode, resnet_num=resnet_num,
                                attention=attentions[cnt], dropout_p=dropout_p))

    def __using_checkpoint(self, x, model):
        x = model(x)
        return x

    def forward(self, x, return_intermediates=False):
        feature = [x]
        for layer in self.layer:
            if self.checkpoint_enable:
                x = checkpoint(
                    self.__using_checkpoint,
                    x, layer,
                    use_reentrant=False
                )
            else:
                x = layer(x)
            if return_intermediates:
                feature.append(x)
        if return_intermediates:
            return feature
        return x


class Decoder(nn.Module):
    def __init__(self, channels, mode="conv", resnet_num=2, attentions=[], dropout_p=.1,
                 checkpoint_enable=False):
        super(Decoder, self).__init__()
        if len(attentions) == 0:
            attentions = [False] * len(channels)
        else:
            if len(attentions) < len(channels):
                attentions = attentions + [False] * (len(channels) - len(attentions))
            else:
                attentions = attentions[:len(channels)]
        self.checkpoint_enable = checkpoint_enable
        self.layer = nn.ModuleList()
        for cnt, item in enumerate(channels):
            self.layer.append(UpSampleBlock(in_channels=item[0], out_channels=item[1], mode=mode, resnet_num=resnet_num,
                                            attention=attentions[cnt], dropout_p=dropout_p))

    def __using_checkpoint(self, x, model):
        x = model(x)
        return x

    def forward(self, x, return_intermediates=False):
        feature = [x]
        for layer in self.layer:
            if self.checkpoint_enable:
                x = checkpoint(
                    self.__using_checkpoint,
                    x, layer,
                    use_reentrant=False
                )
            else:
                x = layer(x)
            if return_intermediates:
                feature.append(x)
        if return_intermediates:
            return feature
        return x
