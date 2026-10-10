from .import_packages import *
from .position_embedding import RotaryEmbedding, LearnablePositionEmbedding

"""
mask:False 为pad ,True为有效, [batch, 1, 1, seq_len]
"""


def get_padding_mask(x, pad_id=0):
    mask = (x != pad_id)
    mask = mask[:, :, None] * mask[:, None, :]
    return mask[:, None, :, :]


def get_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))[None, :, :]


def image2sequences(x):
    b, c, h, w = x.shape
    x = x.reshape(b, c, h * w).transpose(1, 2).contiguous()
    return x, (c, h, w)


def sequences2image(sequences, shape):
    sequences = sequences.transpose(1, 2).reshape(-1, *shape).contiguous()
    return sequences


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                 scale=None) -> torch.Tensor:
    """输入类型:[batch_size, head_num, seq_length, d_k]，且d_model=head_num*d_k"""
    assert key.shape[2] == value.shape[2], f"key和value的维度不同{key.shape[2]}!={value.shape[2]}"
    b, l, s = query.size(0), query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    if len(query.shape) == 3:
        attn_bias = torch.zeros(b, l, s, dtype=query.dtype)  # 输入为(batch, query_seq, d_model)
    else:
        attn_bias = torch.zeros(b, 1, l, s, dtype=query.dtype)  # 输入为(batch, head_num, query_seq, d_model)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(l, s, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    if attn_mask is not None:
        # assert attn_mask.shape[-2:] == (l, s), "掩码形状错误,应为(q_seq_length, k_seq_length)"
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, head_num, dropout_p=0.1, query_key_norm_type="none"):
        super(SelfAttentionBlock, self).__init__()
        assert d_model % head_num == 0, f"d_model需要被head_num均分, 但d_model={d_model}, head_num={head_num}"
        assert query_key_norm_type in ["none", "layer_norm",
                                       "rms_norm"], "query_key_norm_type只支持none, layer_norm, rms_norm三种模式"
        self.dim = d_model // head_num
        self.head_num = head_num
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout_p)
        )
        if query_key_norm_type == "none":
            self.query_norm = nn.Identity()
            self.key_norm = nn.Identity()
        elif query_key_norm_type == "layer_norm":
            self.query_norm = nn.LayerNorm(self.dim)
            self.key_norm = nn.LayerNorm(self.dim)
        else:
            self.query_norm = nn.RMSNorm(self.dim)
            self.key_norm = nn.RMSNorm(self.dim)

    def forward(self, query, mask=None, **kwargs):
        batch_size, seq_len, d_model = query.shape
        qkv = self.qkv(query)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch_size, seq_len, self.head_num, self.dim).transpose(1, 2).contiguous()
        q = self.query_norm(q)
        k = k.reshape(batch_size, seq_len, self.head_num, self.dim).transpose(1, 2).contiguous()
        k = self.key_norm(k)
        v = v.reshape(batch_size, seq_len, self.head_num, self.dim).transpose(1, 2).contiguous()
        score = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_p)
        score = score.transpose(1, 2).reshape(batch_size, seq_len, d_model).contiguous()
        x = self.proj_out(score)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, head_num, dropout_p=0.1, query_key_norm_type="none"):
        super(CrossAttentionBlock, self).__init__()
        assert d_model % head_num == 0, f"d_model需要被head_num均分, 但d_model={d_model}, head_num={head_num}"
        assert query_key_norm_type in ["none", "layer_norm",
                                       "rms_norm"], "query_key_norm_type只支持none, layer_norm, rms_norm三种模式"
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.dropout_p = dropout_p
        self.dim = d_model // head_num
        self.head_num = head_num
        self.d_model = d_model
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout_p)
        )
        if query_key_norm_type == "none":
            self.query_norm = nn.Identity()
            self.key_norm = nn.Identity()
        elif query_key_norm_type == "layer_norm":
            self.query_norm = nn.LayerNorm(self.dim)
            self.key_norm = nn.LayerNorm(self.dim)
        else:
            self.query_norm = nn.RMSNorm(self.dim)
            self.key_norm = nn.RMSNorm(self.dim)

    def forward(self, query, key, value, mask=None):
        batch_size, query_seq_len, d_model = query.shape
        batch_size, key_seq_len, d_model = key.shape
        batch_size, value_seq_len, d_model = value.shape
        q = self.Q(query).reshape(batch_size, query_seq_len, self.head_num, self.dim).transpose(1, 2).contiguous()
        q = self.query_norm(q)
        k = self.K(key).reshape(batch_size, key_seq_len, self.head_num, self.dim).transpose(1, 2).contiguous()
        k = self.key_norm(k)
        v = self.V(value).reshape(batch_size, value_seq_len, self.head_num, self.dim).transpose(1, 2).contiguous()
        score = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_p)
        score = score.transpose(1, 2).reshape(batch_size, query_seq_len, d_model).contiguous()
        x = self.proj_out(score)
        return x


class SelfAttentionBlockWithRotary(nn.Module):
    def __init__(self, d_model, head_num, dropout_p=0.1, max_freq=10, base=10000, query_key_norm_type="none",
                 ):
        super(SelfAttentionBlockWithRotary, self).__init__()
        assert d_model % head_num == 0, f"d_model需要被head_num均分, 但d_model={d_model}, head_num={head_num}"
        assert query_key_norm_type in ["none", "layer_norm",
                                       "rms_norm"], "query_key_norm_type只支持none, layer_norm, rms_norm三种模式"
        self.dim = d_model // head_num
        self.head_num = head_num
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.rotary = RotaryEmbedding(max_freq=max_freq, base=base)
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout_p)
        )
        if query_key_norm_type == "none":
            self.query_norm = nn.Identity()
            self.key_norm = nn.Identity()
        elif query_key_norm_type == "layer_norm":
            self.query_norm = nn.LayerNorm(self.dim)
            self.key_norm = nn.LayerNorm(self.dim)
        else:
            self.query_norm = nn.RMSNorm(self.dim)
            self.key_norm = nn.RMSNorm(self.dim)

    def forward(self, query, mask=None, query_rotary_mode="1d"):
        batch_size, seq_len, d_model = query.shape
        qkv = self.qkv(query)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch_size, seq_len, self.head_num, self.dim).transpose(1, 2).contiguous()
        q = self.rotary(q, mode=query_rotary_mode)
        q = self.query_norm(q)
        k = k.reshape(batch_size, seq_len, self.head_num, self.dim).transpose(1, 2).contiguous()
        k = self.rotary(k, mode=query_rotary_mode)
        k = self.key_norm(k)
        v = v.reshape(batch_size, seq_len, self.head_num, self.dim).transpose(1, 2).contiguous()
        score = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_p)
        score = score.transpose(1, 2).reshape(batch_size, seq_len, d_model).contiguous()
        x = self.proj_out(score)
        return x


class ImageSelfAttentionBlockWithRotary(nn.Module):
    def __init__(self, channels, max_freq=10, dropout_p=.1, query_key_norm_type="none"):
        super().__init__()
        assert query_key_norm_type in ["none", "layer_norm",
                                       "rms_norm"], "query_key_norm_type只支持none, layer_norm, rms_norm三种模式"
        self.qkv = nn.Conv2d(in_channels=channels, out_channels=channels * 3, kernel_size=3, stride=1, padding=1)
        self.dropout_p = dropout_p
        self.channels = channels
        if channels % 64 == 0:
            head_num = max(channels // 64, 1)
        else:
            head_num = 8
            while channels % head_num != 0:
                head_num -= 1
        self.head_num = head_num
        self.dim = channels // head_num
        self.rotary = RotaryEmbedding(max_freq=max_freq)
        self.proj_out = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        if query_key_norm_type == "none":
            self.query_norm = nn.Identity()
            self.key_norm = nn.Identity()
        elif query_key_norm_type == "layer_norm":
            self.query_norm = nn.LayerNorm(self.dim)
            self.key_norm = nn.LayerNorm(self.dim)
        else:
            self.query_norm = nn.RMSNorm(self.dim)
            self.key_norm = nn.RMSNorm(self.dim)

    def forward(self, query, mask=None):
        batch_size, c, h, w = query.shape
        qkv = self.qkv(query)
        q, k, v = qkv.chunk(3, dim=1)
        q, shape = image2sequences(q)
        q = q.reshape(batch_size, -1, self.head_num, self.dim).transpose(1, 2).contiguous()
        q = self.rotary(q, mode="nd", shape=shape[1:])
        q = self.query_norm(q)
        k, *_ = image2sequences(k)
        k = k.reshape(batch_size, -1, self.head_num, self.dim).transpose(1, 2).contiguous()
        k = self.rotary(k, mode="nd", shape=shape[1:])
        k = self.key_norm(k)
        v, *_ = image2sequences(v)
        v = v.reshape(batch_size, -1, self.head_num, self.dim).transpose(1, 2).contiguous()
        score = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_p)
        score = score.transpose(1, 2).reshape(batch_size, -1, self.channels)
        score = sequences2image(score, shape=shape)
        x = self.proj_out(score)
        return x


class ImageCrossAttentionBlock(nn.Module):
    def __init__(self, channels, key_dim, dropout_p=.1, query_key_norm_type="none", max_freq=10):
        super().__init__()
        assert query_key_norm_type in ["none", "layer_norm",
                                       "rms_norm"], "query_key_norm_type只支持none, layer_norm, rms_norm三种模式"
        self.Q = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.K = nn.Linear(key_dim, channels)
        self.V = nn.Linear(key_dim, channels)
        self.dropout_p = dropout_p
        self.channels = channels
        self.rotary = RotaryEmbedding(max_freq=max_freq)
        if channels % 64 == 0:
            head_num = max(channels // 64, 1)
        else:
            head_num = 8
            while channels % head_num != 0:
                head_num -= 1
        self.head_num = head_num
        self.dim = channels // head_num
        self.proj_out = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        if query_key_norm_type == "none":
            self.query_norm = nn.Identity()
            self.key_norm = nn.Identity()
        elif query_key_norm_type == "layer_norm":
            self.query_norm = nn.LayerNorm(self.dim)
            self.key_norm = nn.LayerNorm(self.dim)
        else:
            self.query_norm = nn.RMSNorm(self.dim)
            self.key_norm = nn.RMSNorm(self.dim)

    def forward(self, query, key, value, mask=None):
        batch_size, c, h, w = query.shape
        q = self.Q(query)
        k = self.K(key)
        v = self.V(value)
        q, shape = image2sequences(q)
        q = q.reshape(batch_size, -1, self.head_num, self.dim).transpose(1, 2).contiguous()
        q = self.rotary(q, mode="nd", shape=shape[1:])
        q = self.query_norm(q)
        k = k.reshape(batch_size, -1, self.head_num, self.dim).transpose(1, 2).contiguous()
        k = self.key_norm(k)
        v = v.reshape(batch_size, -1, self.head_num, self.dim).transpose(1, 2).contiguous()
        score = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_p)
        score = score.transpose(1, 2).reshape(batch_size, -1, self.channels)
        score = sequences2image(score, shape=shape)
        x = self.proj_out(score)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p=.1):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.layer(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, head_num, mlp_ratio=2, dropout_p=.1, query_key_norm_type="rms_norm", max_freq=10,
                 base=10000, rotary=False):
        super(TransformerEncoderBlock, self).__init__()
        if rotary:
            self.self_attn = SelfAttentionBlockWithRotary(d_model=d_model, head_num=head_num, dropout_p=dropout_p,
                                                          max_freq=max_freq, base=base,
                                                          query_key_norm_type=query_key_norm_type,
                                                          )
        else:
            self.self_attn = SelfAttentionBlock(d_model=d_model, head_num=head_num, dropout_p=dropout_p,
                                                query_key_norm_type=query_key_norm_type)
        self.ffn = FeedForward(d_model=d_model, dropout_p=dropout_p, d_ff=d_model * mlp_ratio)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x, mask=None, query_rotary_mode="1d"):
        h_ = self.norm1(x)
        x = x + self.self_attn(h_, mask=mask, query_rotary_mode=query_rotary_mode)
        h_ = self.norm2(x)
        x = x + self.ffn(h_)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, layer_num, d_model, head_num, mlp_ratio=2, dropout_p=.1, query_key_norm_type="rms_norm",
                 max_freq=10,
                 base=10000, rotary=False, checkpoint_enable=False):
        super(TransformerEncoderLayer, self).__init__()
        self.checkpoint_enable = checkpoint_enable
        self.layer = nn.ModuleList()
        for i in range(layer_num):
            self.layer.append(
                TransformerEncoderBlock(d_model=d_model, head_num=head_num, mlp_ratio=mlp_ratio, dropout_p=dropout_p,
                                        query_key_norm_type=query_key_norm_type, max_freq=max_freq, base=base,
                                        rotary=rotary))

    def __using_checkpoint(self, x, mask, query_rotary_mode, model):
        x = model(x, mask, query_rotary_mode)
        return x

    def forward(self, x, mask=None, query_rotary_mode="1d", return_intermediates=False):
        feature = []
        for layer in self.layer:
            if self.checkpoint_enable:
                x = checkpoint(
                    self.__using_checkpoint,
                    x, mask, query_rotary_mode, layer,
                    use_reentrant=False
                )
            else:
                x = layer(x, mask=mask, query_rotary_mode=query_rotary_mode)
            if return_intermediates:
                feature.append(x)
        if return_intermediates:
            return feature
        return x
