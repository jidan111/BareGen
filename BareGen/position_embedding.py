from .import_packages import *


@autocast(device_type="cuda", enabled=False)
def rotate_half_neox(x):
    assert x.shape[-1] % 2 == 0, "维度需要被均分2份"
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


@autocast(device_type="cuda", enabled=False)
def apply_rope_neox(x, sin_, cos_):
    """
    x: [batch, head, seq, dim]
    sin_:[seq, dim//2]
    cos_:[seq, dim//2]
    """
    sin_ = sin_.repeat(1, 2)  # [seq, dim]
    cos_ = cos_.repeat(1, 2)  # [seq, dim]
    return x * cos_[None, None, :, :] + rotate_half_neox(x) * sin_[None, None, :, :]


@autocast(device_type="cuda", enabled=False)
def apply_neox_rope_nodup(x, sin_, cos_):
    """
    x: [batch, head, seq, dim]
    sin_:[seq, dim//2]
    cos_:[seq, dim//2]
    """
    x1, x2 = x.chunk(2, dim=-1)
    o1 = x1 * cos_ - x2 * sin_
    o2 = x2 * cos_ + x1 * sin_
    return torch.cat([o1, o2], dim=-1)


@autocast(device_type="cuda", enabled=False)
def get_sin_cos_1d(cur_pos=2048, dim=64, base=10000, device="cpu", inv_freq=None):
    assert dim % 2 == 0, "维度需要被均分为实部和虚部"
    if inv_freq is None:
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)).unsqueeze(
            0)
    else:
        inv_freq = inv_freq.unsqueeze(0)
    pos = torch.arange(cur_pos, dtype=torch.float32, device=device).unsqueeze(1)
    phi = pos * inv_freq
    return phi.sin(), phi.cos()


@autocast(device_type="cuda", enabled=False)
def get_sin_cos_nd(shape=(32, 32), max_freq=10, dim=64, device="cpu", inv_freq=None):
    n_slots_total = dim // 2
    num_axis = len(shape)
    # 总slot必须可以平分给每一个轴
    assert n_slots_total % num_axis == 0, f"总slot(dim//2={n_slots_total})必须可以被轴数{num_axis}整除"
    per_axis_slot = n_slots_total // num_axis
    if inv_freq is None:
        inv_freq = (
                torch.linspace(1., max_freq / 2, per_axis_slot, device=device, dtype=torch.float32)
                * math.pi
        ).unsqueeze(0)  # [1, per_axis_slot]
    else:
        inv_freq = inv_freq.unsqueeze(0)
    phi_list = []
    for idx, axis_len in enumerate(shape):
        pos = torch.linspace(-1.0, 1.0, steps=axis_len, device=device, dtype=torch.float32).unsqueeze(
            1)  # [axis_len, 1]
        phi = pos * inv_freq
        # 扩维，把当前轴放到对应维度，其余维度为None
        reshape_shape = [1] * num_axis
        reshape_shape[idx] = axis_len
        reshape_shape.append(per_axis_slot)
        phi = phi.reshape(reshape_shape)
        phi_list.append(phi)
    target_shape = list(shape) + [per_axis_slot]
    phi_expanded = [t.expand(*target_shape) for t in phi_list]
    phi_grid = torch.cat(phi_expanded, dim=-1)
    seq_len = math.prod(shape)
    phi_slot = phi_grid.reshape(seq_len, n_slots_total)
    sin_ = torch.sin(phi_slot)
    cos_ = torch.cos(phi_slot)
    return sin_, cos_


class RotaryEmbedding(nn.Module):
    def __init__(self, max_freq=10, base=10000):
        super(RotaryEmbedding, self).__init__()
        self.max_freq = max_freq
        self.base = base

    @autocast(device_type="cuda", enabled=False)
    def get_sin_cos(self, cur_pos=77, mode="1d", dim=64, shape=None, device="cpu"):
        if mode == "1d":
            sin_, cos_ = get_sin_cos_1d(cur_pos=cur_pos, dim=dim, base=self.base, device=device)
        else:
            sin_, cos_ = get_sin_cos_nd(shape=shape, max_freq=self.max_freq, dim=dim, device=device)
        return sin_, cos_

    @autocast(device_type="cuda", enabled=False)
    def forward(self, x, shape=None, mode="1d"):
        """
        :param x: [batch, head, seq, dim] || [batch, seq, d_model]
        :param shape: tuple
        :param mode: str
        :return:
        """
        if mode == "nd":
            assert shape is not None, "需要显式传入shape参数"
            assert math.prod(shape) == x.shape[-2], "shape模长应该等于序列长度"
        seq_len = x.shape[-2]
        dim = x.shape[-1]
        device = x.device
        sin_, cos_ = self.get_sin_cos(cur_pos=seq_len, mode=mode, dim=dim, shape=shape, device=device)
        return apply_neox_rope_nodup(x, sin_, cos_)


class LearnablePositionEmbedding(nn.Module):
    def __init__(self, shape):
        super(LearnablePositionEmbedding, self).__init__()
        self.weight = nn.Parameter(torch.randn(size=shape))

    def forward(self, x):
        return x + self.weight


def get_sinusoidal_pos_encoding(max_seq_length=2048, d_model=512):
    """生成正余弦位置编码矩阵"""
    pos_encoding = torch.zeros(max_seq_length, d_model)
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-math.log(10000.0) / d_model))
    pos_encoding[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
    pos_encoding[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
    return pos_encoding


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, max_seq_length=2048, d_model=512):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.register_buffer(
            "weight",
            get_sinusoidal_pos_encoding(max_seq_length, d_model)
        )

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.weight[None, :seq_len, :]
