from ..base_structs import *
from .timestep import *


class RectifiedFlow(ConfigModule):
    """
    Rectified Flow: x_t = (1-t)·x_0 + t·x_1, 目标速度场 u = x_1 - x_0
    修复记录:
      6. 时间步嵌入由 Linear(1, dim) 换成正弦嵌入 + MLP (连续时间 t∈[0,1])
      8. 构造期不再绑定 device
    """

    def __init__(self, model: ConfigModule, step_dim=128, frequency_dim=256, max_period=10000.0,
                 nominal_steps=1000.0, **kwargs):
        super(RectifiedFlow, self).__init__()
        # 6. 连续时间步嵌入, step_nums=None 表示输入已在 [0,1]
        self.step_embedding = TimestepEmbedding(dim=step_dim, step_nums=None,
                                                frequency_dim=frequency_dim, max_period=max_period,
                                                nominal_steps=nominal_steps)
        # 8. 构造期不绑定设备
        self.model = model
        self.config["model"] = self.model.config
        self.loss_func = nn.MSELoss()

    @property
    def device(self):
        """8. 设备由参数实际所在位置推断"""
        for param in self.parameters():
            return param.device
        for buffer in self.buffers():
            return buffer.device
        return torch.device("cpu")

    def compute_flow_state(self, x0, x1, t):
        batch_size = t.shape[0]
        t = t.reshape(batch_size, 1, 1, 1)
        return (1 - t) * x0 + t * x1

    def compute_target_velocity(self, x0, x1):
        return x1 - x0

    def generator_time_step(self, batch_size, device=None):
        return torch.rand(size=(batch_size,), device=device if device is not None else self.device)

    def t_embed(self, t):
        """t: [B] float, 取值 [0,1]"""
        return self.step_embedding(t)

    def forward(self, x0, condition1=None, condition2=None, **kwargs):
        batch_size = x0.shape[0]
        x1 = torch.randn_like(x0)
        t = self.generator_time_step(batch_size=batch_size, device=x0.device)
        xt = self.compute_flow_state(x0=x0, x1=x1, t=t)
        ut = self.compute_target_velocity(x0=x0, x1=x1)
        t_embed = self.t_embed(t)
        vt = self.model(xt, t_embed, condition1, condition2, **kwargs)
        loss = self.loss_func(vt, ut)
        return loss

    def loss(self, x, condition1=None, condition2=None, **kwargs):
        loss = self(x, condition1, condition2, **kwargs)
        return loss

    @torch.no_grad()
    def sample_euler(self, x, step_nums=100, condition1=None, condition2=None, **kwargs):
        batch_size = x.shape[0]
        dt = -1. / step_nums
        steps = torch.linspace(1.0, 0.0, step_nums + 1, device=x.device)
        for i in tqdm(range(step_nums), desc="Euler Sampling"):
            t = steps[i].expand(size=(batch_size,))
            vt = self.model(x, self.t_embed(t), condition1, condition2, **kwargs)
            x = x + dt * vt
        return x

    @torch.no_grad()
    def sample_rk4(self, x, step_nums=100, condition1=None, condition2=None, **kwargs):
        def model_predict(x_in, t_in, condition1_in, condition2_in):
            return self.model(x_in, self.t_embed(t_in), condition1_in, condition2_in, **kwargs)

        batch_size = x.shape[0]
        dt = -1. / step_nums
        steps = 1. - torch.arange(start=0, end=step_nums, step=1, device=x.device) / step_nums
        for i in tqdm(range(step_nums), desc="Rk4 Sampling"):
            t = steps[i].expand(size=(batch_size,))
            f1 = model_predict(x_in=x, t_in=t, condition1_in=condition1, condition2_in=condition2)
            f2 = model_predict(x_in=x + 0.5 * dt * f1, t_in=t + 0.5 * dt, condition1_in=condition1,
                               condition2_in=condition2)
            f3 = model_predict(x_in=x + 0.5 * dt * f2, t_in=t + 0.5 * dt, condition1_in=condition1,
                               condition2_in=condition2)
            f4 = model_predict(x_in=x + dt * f3, t_in=t + dt, condition1_in=condition1, condition2_in=condition2)
            x = x + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
        return x

    @torch.no_grad()
    def sample(self, batch_size, image_shape, x=None, condition1=None, condition2=None, step_nums=100, mode="euler",
               **kwargs):
        assert mode in ["euler", "rk4"], f"支持 euler/rk4 采样, 但传入 {mode}"
        if x is None:
            x = torch.randn(size=(batch_size, *image_shape), device=self.device)
        if mode == "euler":
            return self.sample_euler(x=x, step_nums=step_nums, condition1=condition1, condition2=condition2, **kwargs)
        return self.sample_rk4(x=x, step_nums=step_nums, condition1=condition1, condition2=condition2, **kwargs)
