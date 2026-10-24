from ..attention import *
from ..position_embedding import *
from ..base_structs import *


class TextTransformer(ConfigModule):
    def __init__(self, vocab_size, out_dim=512, max_seq_len=4096, layer_num=8, d_model=512, head_num=8, mlp_ratio=4,
                 dropout_p=0.1, query_key_norm_type="rms_norm", using_rotary_pe=True, base=10000,
                 checkpoint_enable=False, pad_id=0):
        super(TextTransformer, self).__init__()
        self.pad_id = pad_id
        self.vocab_embed = nn.Embedding(vocab_size, d_model)
        if using_rotary_pe:
            self.pe = IdentityContinue()
        else:
            self.pe = SinusoidalPositionEmbedding(max_seq_length=max_seq_len, d_model=d_model)
        self.init_linear = nn.Linear(d_model, d_model)
        self.transformer = TransformerEncoderLayer(layer_num=layer_num, d_model=d_model, head_num=head_num,
                                                   mlp_ratio=mlp_ratio,
                                                   dropout_p=dropout_p,
                                                   query_key_norm_type=query_key_norm_type,
                                                   max_freq=10,
                                                   base=base, rotary=using_rotary_pe,
                                                   checkpoint_enable=checkpoint_enable)
        self.norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, out_dim)

    def forward(self, tokens):
        """
        :param tokens:</start> token1, token2 ... </end> </pad> ... </pad>
        </end>为词表最大索引, </pad>为词表0索引
        """
        batch_size = tokens.shape[0]
        mask = get_padding_mask(tokens, pad_id=self.pad_id).to(tokens.device)
        x = self.vocab_embed(tokens)
        x = self.pe(x)
        x = self.init_linear(x)
        x = self.transformer(x, mask=mask)
        x = self.norm(x)
        x = x[torch.arange(batch_size), tokens.argmax(-1)]
        x = self.proj_out(x)
        return x

    @torch.no_grad()
    def encode_text(self, tokens):
        batch_size = tokens.shape[0]
        mask = get_padding_mask(tokens, pad_id=self.pad_id).to(tokens.device)
        x = self.vocab_embed(tokens)
        x = self.pe(x)
        x = self.init_linear(x)
        x = self.transformer(x, mask=mask, query_rotary_mode="1d")
        global_embed = self.norm(x)
        pool_embed = global_embed[torch.arange(batch_size), tokens.argmax(-1)]
        pool_embed = self.proj_out(pool_embed)
        return global_embed, pool_embed

    def loss(self):
        pass
