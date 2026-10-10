from ..attention import *
from ..base_structs import *


class ViTSelfAttentionBlockWithRotary(SelfAttentionBlockWithRotary):
    def rotary_query_or_key(self, x, image_shape):
        cls_token = x[:, :, 0, :]
        image_token = x[:, :, 1:, :]
        image_token = self.rotary(image_token, mode="nd", shape=image_shape)
        x = torch.cat([cls_token[:, :, None, :], image_token], dim=2).contiguous()
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


class ViTTransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, head_num, mlp_ratio=2, dropout_p=.1, query_key_norm_type="rms_norm", max_freq=10,
                 base=10000, rotary=False):
        super(ViTTransformerEncoderBlock, self).__init__()
        if rotary:
            self.self_attn = ViTSelfAttentionBlockWithRotary(d_model=d_model, head_num=head_num, dropout_p=dropout_p,
                                                             max_freq=max_freq, base=base,
                                                             query_key_norm_type=query_key_norm_type)
        else:
            self.self_attn = SelfAttentionBlock(d_model=d_model, head_num=head_num, dropout_p=dropout_p,
                                                query_key_norm_type=query_key_norm_type)
        self.ffn = FeedForward(d_model=d_model, dropout_p=dropout_p, d_ff=d_model * mlp_ratio)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x, mask=None, image_shape=None):
        h_ = self.norm1(x)
        x = x + self.self_attn(h_, mask=mask, image_shape=image_shape)
        h_ = self.norm2(x)
        x = x + self.ffn(h_)
        return x


class ViTTransformerEncoderLayer(nn.Module):
    def __init__(self, layer_num, d_model, head_num, mlp_ratio=2, dropout_p=.1, query_key_norm_type="rms_norm",
                 max_freq=10,
                 rotary=False, checkpoint_enable=False):
        super(ViTTransformerEncoderLayer, self).__init__()
        self.checkpoint_enable = checkpoint_enable
        self.layer = nn.ModuleList()
        for i in range(layer_num):
            self.layer.append(
                ViTTransformerEncoderBlock(d_model=d_model, head_num=head_num, mlp_ratio=mlp_ratio, dropout_p=dropout_p,
                                           query_key_norm_type=query_key_norm_type, max_freq=max_freq,
                                           rotary=rotary))

    def __using_checkpoint(self, x, mask, image_shape, model):
        x = model(x, mask, image_shape)
        return x

    def forward(self, x, mask=None, image_shape=None, return_intermediates=False):
        feature = []
        for layer in self.layer:
            if self.checkpoint_enable:
                x = checkpoint(
                    self.__using_checkpoint,
                    x, mask, image_shape, layer,
                    use_reentrant=False
                )
            else:
                x = layer(x, mask=mask, image_shape=image_shape)
            if return_intermediates:
                feature.append(x)
        if return_intermediates:
            return feature
        return x


class VisionTransformer(ConfigModule):
    def __init__(self, in_channels, out_dim, d_model, patch_size, layer_num, head_num, dropout_p=.1,
                 max_freq=10, checkpoint_enable=False, mlp_ratio=4, query_key_norm_type="rms_norm",
                 using_rotary_pe=True, image_shape=None):
        super(VisionTransformer, self).__init__()
        if not using_rotary_pe:
            assert image_shape is not None, "固定位置编码，必须显示输入image_shape"
        if not using_rotary_pe:
            h_p, w_p = image_shape[1] // patch_size, image_shape[2] // patch_size
            self.pe = LearnablePositionEmbedding(shape=((h_p * w_p) + 1, d_model))
        else:
            self.pe = IdentityContinue()
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=d_model, kernel_size=patch_size,
                                   stride=patch_size)
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.transformer = ViTTransformerEncoderLayer(layer_num=layer_num, d_model=d_model, head_num=head_num,
                                                      mlp_ratio=mlp_ratio,
                                                      dropout_p=dropout_p,
                                                      query_key_norm_type=query_key_norm_type,
                                                      max_freq=max_freq,
                                                      rotary=using_rotary_pe,
                                                      checkpoint_enable=checkpoint_enable)
        self.norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, out_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.init_conv(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(size=(batch_size, 1, self.d_model))
        x = torch.cat((cls_token, x), dim=1).contiguous()
        x = self.pe(x)
        x = self.transformer(x, image_shape=(h, w))
        x = self.norm(x)
        pool_token = x[:, 0, :]
        pool_token = self.proj_out(pool_token)
        return pool_token

    def loss(self):
        pass
