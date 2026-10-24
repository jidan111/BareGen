from ..import_packages import *
from ..attention import *


class DiagonalGaussianDistribution(object):
    def __init__(self, tensor, deterministic=False):
        super(DiagonalGaussianDistribution, self).__init__()
        assert tensor.shape[1] % 2 == 0, f"输入的潜在向量无法划分为均值和方差, {tensor.shape[1]}%2 != 0"
        self.dim = list(range(1, len(tensor.shape)))
        self.params = tensor
        self.mean, self.log_var = tensor.chunk(2, dim=1)
        self.log_var = self.log_var.clamp(-30.0, 20.0)
        self.deterministic = deterministic
        if deterministic:
            self.var, self.std = torch.zeros_like(self.mean)
        else:
            self.std = torch.exp(0.5 * self.log_var)
            self.var = self.log_var.exp()

    def sample(self):
        out = self.mean + self.std * torch.randn_like(self.mean).to(self.params.device)
        return out

    def mode(self):
        return self.mean

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(self.mean.pow(2) + self.var - 1. - self.log_var, dim=self.dim)
            else:
                return 0.5 * torch.sum((self.mean - other.mean).pow(
                    2) / other.var + self.var / other.var - 1. - self.log_var + other.logvar, dim=self.dim)

    def nll(self, sample):
        if self.deterministic:
            return torch.Tensor([0.])
        logwopi = np.log(2. * np.pi)
        return 0.5 * torch.sum(logwopi + self.log_var + (sample - self.mean).pow(2) / self.var, dim=self.dim)


class VQBridgeProjectorAttention(nn.Module):
    def __init__(self, embed_num, embed_dim, head_nums=4, layer_num=2, dropout_p=.1, mul_size=16):
        super(VQBridgeProjectorAttention, self).__init__()
        self.d_model = embed_dim * mul_size
        self.embed_dim = embed_dim
        self.embed_num = embed_num
        self.init = nn.Linear(embed_dim, self.d_model)
        self.out = nn.Linear(self.d_model, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(size=(1, embed_num, embed_dim)))
        self.vit = TransformerEncoderLayer(layer_num=layer_num, d_model=self.d_model, head_num=head_nums, mlp_ratio=2,
                                           dropout_p=dropout_p,
                                           query_key_norm_type="rms_norm",
                                           max_freq=10,
                                           base=10000, rotary=True)

    def forward(self, codebook):
        codebook = codebook.unsqueeze(0)  # [1, N, D]
        codebook = codebook + self.pos_embed  # [1, N, D]
        codebook = self.init(codebook)
        codebook = self.vit(codebook)  # [1, N, D]
        codebook = self.out(codebook)
        codebook = codebook.squeeze(0)  # [N, D]
        return codebook


class VQBridgeProjectorMLP(nn.Module):
    def __init__(self, embed_num, embed_dim, vq_bridge_layer_num):
        super(VQBridgeProjectorMLP, self).__init__()
        vq_bridge_layer_num = max(2, vq_bridge_layer_num)
        self.embed_dim = embed_dim
        self.embed_num = embed_num
        self.pos_embed = nn.Parameter(torch.randn(size=(1, embed_num, embed_dim)))
        layer = []
        for i in range(vq_bridge_layer_num - 1):
            layer.extend([nn.Linear(embed_dim, embed_dim),
                          nn.SiLU()])
        layer.append(nn.Linear(embed_dim, embed_dim))
        self.layer = nn.Sequential(*layer)

    def forward(self, codebook):
        codebook = codebook.unsqueeze(0)  # [1, N, D]
        codebook = codebook + self.pos_embed  # [1, N, D]
        codebook = self.layer(codebook)
        codebook = codebook.squeeze(0)  # [N, D]
        return codebook


class VectorQuantizer(nn.Module):
    def __init__(self, embed_num, embed_dim, beta=0.25,
                 use_vq_bridge=True, vq_bridge_mul_size=16, vq_bridge_head_num=4, vq_bridge_layer_num=2,
                 vq_bridge_dropout_p=.1, vq_bridge_type="attn"):
        super(VectorQuantizer, self).__init__()
        assert vq_bridge_type in ["attn", "mlp"], "只支持attn和mlp两种bridge方式"
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.beta = beta
        self.use_vq_bridge = use_vq_bridge
        self.embedding = nn.Embedding(num_embeddings=self.embed_num, embedding_dim=self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.embed_num, 1.0 / self.embed_num)
        if use_vq_bridge:
            if vq_bridge_type == "attn":
                self.vq_bridge = VQBridgeProjectorAttention(
                    embed_num=embed_num,
                    embed_dim=embed_dim,
                    head_nums=vq_bridge_head_num,
                    layer_num=vq_bridge_layer_num,
                    dropout_p=vq_bridge_dropout_p,
                    mul_size=vq_bridge_mul_size
                )
            else:
                self.vq_bridge = VQBridgeProjectorMLP(embed_dim=embed_dim, embed_num=embed_num,
                                                      vq_bridge_layer_num=vq_bridge_layer_num)

    def forward(self, z):
        optimized_codebook = self.embedding.weight
        if self.use_vq_bridge:
            optimized_codebook = self.vq_bridge(optimized_codebook)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embed_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(optimized_codebook ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, optimized_codebook.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = optimized_codebook[min_encoding_indices].reshape(z.shape)
        # ===== 3. 损失函数改造 =====
        # 1) Commitment loss：约束编码器向优化码本靠拢
        commitment_loss = torch.mean((z_q.detach() - z) ** 2)
        # 2) Codebook loss：约束优化码本向编码器输出靠拢
        codebook_loss = self.beta * torch.mean((z_q - z.detach()) ** 2)
        # 3) VQBridge正则化：防止优化码本偏离原始分布
        # vq_bridge_reg = 0.01 * F.mse_loss(optimized_codebook, self.embedding.weight)
        # total_loss = commitment_loss + codebook_loss + vq_bridge_reg
        total_loss = commitment_loss + codebook_loss
        # STE直通估计（保持离散特性）
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, total_loss, min_encoding_indices

    def get_codebook_entry(self, indices, shape, optimized_codebook=None):
        if optimized_codebook is None:
            if self.use_vq_bridge:
                optimized_codebook = self.vq_bridge(self.embedding.weight)
            else:
                optimized_codebook = self.embedding.weight
        z_q = optimized_codebook[indices]
        z_q = z_q.view(shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    @torch.no_grad()
    def feature2latent(self, z):
        batch_size, c, h, w = z.shape
        optimized_codebook = self.embedding.weight
        if self.use_vq_bridge:
            optimized_codebook = self.vq_bridge(optimized_codebook)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embed_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(optimized_codebook ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, optimized_codebook.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        latent = self.get_codebook_entry(indices=min_encoding_indices, shape=(batch_size, h, w, c),
                                         optimized_codebook=optimized_codebook)
        return latent
