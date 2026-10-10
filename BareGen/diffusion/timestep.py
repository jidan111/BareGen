from ..import_packages import *


@autocast(device_type="cuda", enabled=False)
def sinusoidal_timestep_embedding(t, dim, max_period=10000.0):
    """
    正弦(傅立叶)时间步嵌入
    :param t: [B] float, 已换算到"名义步数"尺度(如 0~1000)
    :param dim: 嵌入维度, 必须为偶数
    :param max_period: 最长周期, 越大低频分量越多
    :return: [B, dim]
    """
    assert dim % 2 == 0, f"dim需要被2整除, 但dim={dim}"
    half = dim // 2
    device = t.device
    freq = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=device) / half
    )  # [half,]
    args = t.float().unsqueeze(-1) * freq.unsqueeze(0)  # [N,1]*[1, half]=[N,half]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [N, dim]


class TimestepEmbedding(nn.Module):
    """
    正弦时间步嵌入 + MLP，用于替换 nn.Embedding(step_nums, step_dim)。

    相比 nn.Embedding 的优势:
      1. 可外推到训练范围之外的时间步，支持连续时间(如 RectifiedFlow 的 t∈[0,1])
      2. 修改 step_nums 不再改变参数形状，旧 checkpoint 不会因为改步数而失效
      3. 低频/高频分量齐全，表达力强于纯 Linear(1, dim)

    :param dim: 输出维度
    :param step_nums: 离散步数；传 None 表示输入已经是 [0,1] 的连续时间
    :param frequency_dim: 正弦嵌入维度
    :param max_period: 正弦最长周期
    :param nominal_steps: 归一化后的名义步数尺度(对齐 DDPM 的 T=1000 惯例)
    :param hidden_dim: MLP 隐层维度, 默认等于 dim
    """

    def __init__(self, dim, step_nums=None, frequency_dim=256, max_period=10000.0,
                 nominal_steps=1000.0, hidden_dim=None):
        super(TimestepEmbedding, self).__init__()
        assert frequency_dim % 2 == 0, f"frequency_dim需要被2整除, 但frequency_dim={frequency_dim}"
        self.dim = dim
        self.step_nums = step_nums
        self.frequency_dim = frequency_dim
        self.max_period = max_period
        self.nominal_steps = float(nominal_steps)
        hidden_dim = hidden_dim if hidden_dim is not None else dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t):
        """
        :param t: [B] long(离散步数, 0~step_nums-1) 或 [B] float(连续时间, [0,1])
        :return: [B, dim]
        """
        t = t.reshape(-1).to(torch.float32)
        if self.step_nums is not None:
            # 离散步数 -> 名义步数尺度
            t = t * (self.nominal_steps / max(float(self.step_nums - 1), 1.0))
        else:
            # 连续时间默认已在 [0,1]
            t = t * self.nominal_steps
        emb = sinusoidal_timestep_embedding(t, dim=self.frequency_dim, max_period=self.max_period)
        # 正弦部分强制 fp32 计算，这里对齐到 MLP 权重类型，避免 bf16/fp16 下的类型冲突
        emb = emb.to(self.mlp[0].weight.dtype)
        return self.mlp(emb)


class LearnableTimestepEmbedding(nn.Module):
    def __init__(self, dim, step_nums=None):
        super(LearnableTimestepEmbedding, self).__init__()
        self.weight = nn.Embedding(step_nums, dim)

    def forward(self, x):
        return self.weight(x)


class MLPTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super(MLPTimestepEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim*2),
            nn.SiLU(),
            nn.Linear(dim*2, dim),
        )

    def forward(self, x):
        x = x[:, None]
        return self.mlp(x)
