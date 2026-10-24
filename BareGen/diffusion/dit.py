from ..base_structs import *


def modulate_dit(x, shift, scale):
    """
    x * (1 + scale) + shift
    x: [B,S,D]
    shift:[B,D]
    scale:[B,D]
    """
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTSelfAttentionBlockWithRotary(SelfAttentionBlockWithRotary):
    def rotary_query_or_key(self, x, image_shape):
        split_index = math.prod(image_shape)
        if split_index == x.shape[2]:
            return self.rotary(x, mode="nd", shape=image_shape)
        image_token = x[:, :, :split_index, :]
        text_token = x[:, :, split_index:, :]
        image_token = self.rotary(image_token, mode="nd", shape=image_shape)
        text_token = self.rotary(text_token, mode="1d")
        x = torch.cat([image_token, text_token], dim=2).contiguous()
        return x

    def forward(self, query, mask=None, image_shape=None):
        batch_size, seq_len, d_model = query.shape
        qkv = self.qkv(query)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch_size, seq_len, self.head_num, self.dim).transpose(1, 2).contiguous()
        q = self.rotary_query_or_key(q, image_shape=image_shape)
        q = self.query_norm(q)
        k = k.reshape(batch_size, seq_len, self.head_num, self.dim).transpose(1, 2).contiguous()
        k = self.rotary_query_or_key(k, image_shape=image_shape)
        k = self.key_norm(k)
        v = v.reshape(batch_size, seq_len, self.head_num, self.dim).transpose(1, 2).contiguous()
        score = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_p)
        score = score.transpose(1, 2).reshape(batch_size, seq_len, d_model).contiguous()
        x = self.proj_out(score)
        return x


class DiTBlock(nn.Module):
    def __init__(self, d_model, condition_dim, head_num, dropout_p=.1, base=10000,
                 query_key_norm_type="none", max_freq=10, mlp_ratio=4, using_rotary_pe=True):
        super(DiTBlock, self).__init__()
        self.attn_norm = nn.RMSNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp_norm = nn.RMSNorm(d_model, elementwise_affine=False, eps=1e-6)
        if using_rotary_pe:
            self.self_attn = DiTSelfAttentionBlockWithRotary(d_model=d_model, head_num=head_num,
                                                             dropout_p=dropout_p,
                                                             max_freq=max_freq,
                                                             base=base,
                                                             query_key_norm_type=query_key_norm_type)
        else:
            self.self_attn = SelfAttentionBlock(d_model=d_model, head_num=head_num, dropout_p=dropout_p,
                                                query_key_norm_type=query_key_norm_type)
        self.mlp = FeedForward(d_model=d_model, dropout_p=dropout_p, d_ff=d_model * mlp_ratio)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 6 * d_model)
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, condition, image_shape=None, mask=None):
        """
        x: [B,S,D]
        condition: [B, D]
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(condition).chunk(6, dim=1)
        x_norm1 = self.attn_norm(x)
        x_norm1 = modulate_dit(x_norm1, shift_msa, scale_msa)
        attn = self.self_attn(x_norm1, image_shape=image_shape, mask=mask)
        x = x + gate_msa[:, None, :] * attn
        x_norm2 = self.attn_norm(x)
        x_norm2 = modulate_dit(x_norm2, shift_mlp, scale_mlp)
        mlp = self.mlp(x_norm2)
        x = x + gate_mlp[:, None, :] * mlp
        return x


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


class DiT(ConfigModule):
    def __init__(self, in_channels, d_model, patch_size, timestep_dim, condition_dim, layer_num, head_num, dropout_p=.1,
                 base=10000, max_freq=10, checkpoint_enable=False, mlp_ratio=4, query_key_norm_type="rms_norm",
                 using_rotary_pe=True, image_shape=None):
        super(DiT, self).__init__()
        token_dim = in_channels * patch_size * patch_size
        assert token_dim <= d_model, f"图片转换token维度({in_channels}*{patch_size}*{patch_size})小于d_model({d_model})， 模型收敛困难"
        if not using_rotary_pe:
            assert image_shape is not None, "固定位置编码，必须显示输入image_shape"
        if not using_rotary_pe:
            h_p, w_p = image_shape[1] // patch_size, image_shape[2] // patch_size
            self.pe = LearnablePositionEmbedding(shape=(h_p * w_p, d_model))
        else:
            self.pe = IdentityContinue()
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=d_model, kernel_size=patch_size,
                                   stride=patch_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.condition_flag = condition_dim is not None
        self.checkpoint_enable = checkpoint_enable
        if condition_dim is not None:
            self.timestep_ln = AdaTimestepNorm(dim=timestep_dim, condition_dim=condition_dim)
            if condition_dim != d_model:
                self.condition_exchange = nn.Linear(condition_dim, d_model)
        else:
            self.timestep_ln = IdentityContinue()
            self.condition_exchange = IdentityContinue()
        self.transformer = nn.ModuleList(
            [DiTBlock(d_model=d_model, condition_dim=timestep_dim, head_num=head_num, dropout_p=dropout_p, base=base,
                      query_key_norm_type=query_key_norm_type, max_freq=max_freq, mlp_ratio=mlp_ratio,
                      using_rotary_pe=using_rotary_pe)
             for _ in range(layer_num)])
        self.post_condition_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timestep_dim, 2 * d_model)
        )
        self.post_norm = nn.RMSNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(d_model, token_dim)
        nn.init.zeros_(self.post_condition_adaLN[-1].weight)
        nn.init.zeros_(self.post_condition_adaLN[-1].bias)

    def __using_checkpoint(self, x, condition, image_shape, mask, model):
        x = model(x, condition, image_shape, mask)
        return x

    def forward(self, x, timestep, condition1, condition2, mask=None, **kwargs):
        timestep = self.timestep_ln(timestep, condition2)
        x = self.init_conv(x)
        batch, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.pe(x)
        if self.condition_flag:
            condition1 = self.condition_exchange(condition1)
            x = torch.cat([x, condition1], dim=1).contiguous()
        for cnt, layer in enumerate(self.transformer):
            if self.checkpoint_enable:
                x = checkpoint(
                    self.__using_checkpoint,
                    x, timestep, (h, w), mask, layer,
                    use_reentrant=False
                )
            else:
                x = layer(x, timestep, (h, w), mask)
        post_scale, post_shift = self.post_condition_adaLN(timestep).chunk(2, dim=1)
        x = self.post_norm(x)
        x = modulate_dit(x, post_scale, post_shift)
        x = self.out(x)
        if self.condition_flag:
            x = x[:, :h * w, :]
        x = x.reshape(batch, h, w, self.in_channels, self.patch_size, self.patch_size).permute(0, 3, 1, 4, 2,
                                                                                               5).reshape(batch,
                                                                                                          self.in_channels,
                                                                                                          h * self.patch_size,
                                                                                                          w * self.patch_size).contiguous()
        return x

    def loss(self):
        pass
