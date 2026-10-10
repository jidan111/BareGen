from ..base_structs import *
from .timestep import *

class Diffusion(ConfigModule):
    """
    DDPM / DDIM / DPM-Solver++(2M) 采样器 + 训练loss
    修复记录:
      1. DDPM 反向方差改用正确的后验方差 β̃_t = β_t·(1-ᾱ_{t-1})/(1-ᾱ_t)，原先误用 √β_t
      2. DDIM 的 σ 项不再重复计入噪声，改为 √(1-ᾱ_{t2}-σ²)·ε + σ·z，并支持 DDIM 的 η 参数
      3. 时间步嵌入由 nn.Embedding 换成正弦嵌入 + MLP，可外推、改步数不破坏 checkpoint
      4. schedule 的 betas 默认值统一为 (1e-4, 0.02)，并对递减的 beta 直接报错
      5. DPM-Solver++(2M) 改为基于 log-SNR(λ) 比值的通用形式，不再假设等间距
      7. loss 支持 SNR 加权 (min_snr / inverse_snr) 与 v-prediction
      8. 构造期不再绑定 device，模型可随意 .to() / 在 CPU 上单测
    """

    def __init__(self, model: ConfigModule, step_nums=100, step_dim=128,
                 schedule_name="linear", betas=(1e-4, 0.02),
                 prediction_type="eps", loss_weighting="min_snr", min_snr_gamma=5.0):
        super(Diffusion, self).__init__()
        assert prediction_type in ["eps", "v"], \
            f"prediction_type只支持 eps/v, 但传入 {prediction_type}"
        assert loss_weighting in [None, "min_snr", "inverse_snr"], \
            f"loss_weighting只支持 None/min_snr/inverse_snr, 但传入 {loss_weighting}"
        self.step_nums = step_nums
        self.prediction_type = prediction_type
        self.loss_weighting = loss_weighting
        self.min_snr_gamma = float(min_snr_gamma)
        self.step_embedding = TimestepEmbedding(dim=step_dim, step_nums=step_nums)
        # self.step_embedding = nn.Embedding(num_embeddings=step_nums, embedding_dim=step_dim)
        self.model = model
        # reduction="none" 是为了按样本做 SNR 加权
        self.loss_func = nn.MSELoss(reduction="none")
        self.config["model"] = self.model.config
        # 调度表完全由 schedule_name/betas 推导，设成非持久化缓冲：
        # 不写进 checkpoint，改 step_nums 后能自动重建，也不会因形状变化加载失败
        for k, v in self.set_params(schedule_name=schedule_name, betas=betas).items():
            self.register_buffer(k, v, persistent=False)

    @property
    def device(self):
        """8. 设备由参数实际所在位置推断，不在 __init__ 里写死"""
        for param in self.parameters():
            return param.device
        for buffer in self.buffers():
            return buffer.device
        return torch.device("cpu")

    def get_alpha_beta(self, schedule_name="linear", betas=(1e-4, 0.02), s=0.008):
        betas = tuple(betas)
        if schedule_name == "linear":
            if betas[0] > betas[1]:
                raise ValueError(
                    f"linear 调度要求 beta 随时间递增, 但传入 betas={betas};"
                    f"若确实需要递减请显式确认后再放开该检查")
            beta = torch.linspace(start=betas[0], end=betas[1], steps=self.step_nums).view(self.step_nums, 1, 1, 1)
            alpha = 1 - beta
            return alpha, beta
        elif schedule_name == "cosine":
            steps = self.step_nums + 1
            x = torch.linspace(0, self.step_nums, steps)
            alpha_bar = torch.cos((x / self.step_nums + s) / (1 + s) * torch.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]  # 归一化
            alpha_bar = alpha_bar[1:]  # 对齐步长
            alpha = alpha_bar[1:] / alpha_bar[:-1]
            alpha = torch.cat([alpha_bar[[0]], alpha])
            beta = 1 - alpha
            beta = torch.clamp(beta, 0.0001, 0.9999)
            beta = beta.reshape(self.step_nums, 1, 1, 1)
            alpha = alpha.reshape(self.step_nums, 1, 1, 1)
            return alpha, beta
        else:
            raise NotImplementedError(
                f"调度算法 {schedule_name} 未预设")

    def set_params(self, schedule_name="linear", betas=(1e-4, 0.02), s=0.008):
        alpha, beta = self.get_alpha_beta(schedule_name=schedule_name, betas=betas, s=s)
        alpha_bar = torch.cumprod(alpha, dim=0)
        # 1. 后验方差需要 ᾱ_{t-1}
        alpha_bar_prev = torch.cat([torch.ones_like(alpha_bar[:1]), alpha_bar[:-1]], dim=0)
        one_sub_alpha_bar = torch.clamp(1 - alpha_bar, min=1e-12)
        one_sub_alpha_bar_prev = torch.clamp(1 - alpha_bar_prev, min=1e-12)
        posterior_variance = torch.clamp(
            beta * one_sub_alpha_bar_prev / one_sub_alpha_bar, min=1e-20
        )
        sqrt_beta = torch.sqrt(beta)
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_sub_alpha_bar = torch.sqrt(one_sub_alpha_bar)
        # 5. λ_t = log(α_t/σ_t) = 0.5·(log ᾱ_t - log(1-ᾱ_t))，DPM-Solver 的 log-SNR
        lambda_t = 0.5 * (torch.log(alpha_bar) - torch.log(one_sub_alpha_bar))
        # 7. SNR_t = ᾱ_t / (1-ᾱ_t)，用于 loss 加权
        return {"sqrt_alpha_bar": sqrt_alpha_bar, "sqrt_one_sub_alpha_bar": sqrt_one_sub_alpha_bar,
                "beta": beta, "sqrt_alpha": sqrt_alpha, "sqrt_beta": sqrt_beta,
                "alpha_bar": alpha_bar, "posterior_variance": posterior_variance, "lambda_t": lambda_t}

    def add_noise(self, x0, noise, t):
        xt = self.sqrt_alpha_bar[t] * x0 + self.sqrt_one_sub_alpha_bar[t] * noise
        return xt

    def _model_predict(self, xt, t, condition1=None, condition2=None, **kwargs):
        """
        统一入口: 模型输出 -> (eps, x0)
        支持 eps 预测与 v 预测 (v = α_t·ε - σ_t·x0)
        """
        t_embed = self.step_embedding(t)
        out = self.model(xt, t_embed, condition1, condition2, **kwargs)
        sqrt_alpha_bar = self.sqrt_alpha_bar[t]
        sqrt_one_sub_alpha_bar = self.sqrt_one_sub_alpha_bar[t]
        if self.prediction_type == "eps":
            eps = out
            x0 = (xt - sqrt_one_sub_alpha_bar * eps) / sqrt_alpha_bar
            return eps, x0
        v = out
        x0 = sqrt_alpha_bar * xt - sqrt_one_sub_alpha_bar * v
        eps = sqrt_one_sub_alpha_bar * xt + sqrt_alpha_bar * v
        return eps, x0

    def _loss_weight(self, t):
        """7. 按时间步的信噪比对 loss 加权, 返回 [B]"""
        alpha_bar = self.alpha_bar[t]
        snr = alpha_bar / torch.clamp(1 - alpha_bar, min=1e-12)
        if self.loss_weighting == "min_snr":
            # Min-SNR-γ: min(SNR, γ) / SNR
            weight = torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma)) / snr
        elif self.loss_weighting == "inverse_snr":
            # 标准 inverse-SNR 加权: w = 1/SNR = (1-ᾱ)/ᾱ
            weight = 1.0 / snr
        else:
            weight = torch.ones_like(snr)
        return weight.reshape(weight.shape[0])

    def forward(self, x, condition1=None, condition2=None, **kwargs):
        batch_size, *_ = x.shape
        t = torch.randint(low=0, high=self.step_nums, size=(batch_size,), dtype=torch.long, device=x.device)
        noise = torch.randn_like(x)
        xt = self.add_noise(x0=x, noise=noise, t=t)
        t_embed = self.step_embedding(t)
        out = self.model(xt, t_embed, condition1, condition2, **kwargs)
        if self.prediction_type == "eps":
            target = noise
        else:
            # 7. v = α_t·ε - σ_t·x0
            target = self.sqrt_alpha_bar[t] * noise - self.sqrt_one_sub_alpha_bar[t] * x
        loss = self.loss_func(out, target)
        loss = loss.reshape(loss.shape[0], -1).mean(dim=-1)
        if self.loss_weighting is not None:
            loss = loss * self._loss_weight(t)
        return loss.mean()

    def loss(self, x, condition1=None, condition2=None, **kwargs):
        loss = self(x, condition1, condition2, **kwargs)
        return loss

    @torch.no_grad()
    def __clean_noise_p(self, xt, t, condition1=None, condition2=None, **kwargs):
        """DDPM 一步去噪"""
        batch_size = xt.shape[0]
        t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=xt.device)
        eps, _ = self._model_predict(xt, t_tensor, condition1, condition2, **kwargs)
        x_t_prev_mean = (xt - (self.beta[t_tensor] / self.sqrt_one_sub_alpha_bar[t_tensor]) * eps) \
            / self.sqrt_alpha[t_tensor]
        if t > 0:
            std = torch.sqrt(self.posterior_variance[t_tensor])
            return x_t_prev_mean + std * torch.randn_like(xt)
        return x_t_prev_mean

    @torch.no_grad()
    def __sample_ddpm(self, batch_size, image_shape, condition1=None, condition2=None, **kwargs):
        x = torch.randn(size=(batch_size, *image_shape), device=self.device)
        for i in tqdm(range(self.step_nums - 1, -1, -1), desc="DDPM Sampling"):
            x = self.__clean_noise_p(x, i, condition1, condition2, **kwargs)
        return x

    @torch.no_grad()
    def __clean_noise_i(self, xt, t1, t2, condition1=None, condition2=None, sigma=0., eta=None, **kwargs):
        """
        DDIM 一步: t1 -> t2 (t2 < t1)
        :param sigma: 直接指定随机项的标准差
        :param eta: DDIM 论文式(16)的 η 参数, 传入时覆盖 sigma
        """
        batch_size = xt.shape[0]
        t1_tensor = torch.full((batch_size,), t1, dtype=torch.long, device=xt.device)
        eps, x0 = self._model_predict(xt, t1_tensor, condition1, condition2, **kwargs)
        if t2 == 0:
            return x0
        alpha_bar_t1 = self.alpha_bar[t1_tensor]
        alpha_bar_t2 = self.alpha_bar[t2]
        if eta is not None:
            # σ_t = η·√((1-ᾱ_{t2})/(1-ᾱ_{t1}))·√(1-ᾱ_{t1}/ᾱ_{t2})
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t2) / (1 - alpha_bar_t1)) \
                * torch.sqrt(torch.clamp(1 - alpha_bar_t1 / alpha_bar_t2, min=0.))
        else:
            sigma_t = torch.full((1, 1, 1), float(sigma), dtype=xt.dtype, device=xt.device)
        # 2. 确定性项要用 √(1-ᾱ_{t2}-σ²)，否则噪声被重复计入
        coef_eps = torch.sqrt(torch.clamp(1 - alpha_bar_t2 - sigma_t ** 2, min=0.))
        return torch.sqrt(alpha_bar_t2) * x0 + coef_eps * eps + sigma_t * torch.randn_like(xt)

    @torch.no_grad()
    def __sample_ddim(self, batch_size, image_shape, x=None, condition1=None, condition2=None, step=2,
                      sigma=0., eta=None, **kwargs):
        steps_arr = self.__get_steps_arr(step)
        if x is None:
            x = torch.randn(size=(batch_size, *image_shape), device=self.device)
        for i in tqdm(range(len(steps_arr) - 1), desc="DDIM Sampling"):
            x = self.__clean_noise_i(x, steps_arr[i], steps_arr[i + 1], condition1, condition2,
                                     sigma=sigma, eta=eta, **kwargs)
        return x

    def __get_steps_arr(self, step):
        """生成降序的时间步序列，并保证以 0 收尾"""
        step = max(int(step), 1)
        steps_arr = list(range(self.step_nums - 1, -1, -step))
        if steps_arr[-1] != 0:
            steps_arr.append(0)
        return steps_arr

    @torch.no_grad()
    def __clean_noise_dpm_2m(self, xt, t1, t2, t1_x0=None, h_prev=None, condition1=None, condition2=None, **kwargs):
        """
        5. DPM-Solver++(2M) 单步: t1 -> t2 (t2 < t1)
        基于 λ_t = log(α_t/σ_t) 的通用形式:
            D̃ = (1 + 1/(2r))·D_i - (1/(2r))·D_{i-1},  r = h_{i-1}/h_i
            x_{t2} = (σ_{t2}/σ_{t1})·x_{t1} + α_{t2}·(1 - e^{-h_i})·D̃
        当 h_i 等间距时 r=1，退化为 (3D - D_prev)/2，与旧实现等价
        :return: (x_t2, x0@t1, h_i)
        """
        batch_size = xt.shape[0]
        t1_tensor = torch.full((batch_size,), t1, dtype=torch.long, device=xt.device)
        _, x0 = self._model_predict(xt, t1_tensor, condition1, condition2, **kwargs)
        if t2 == 0:
            return x0, x0, None
        h = self.lambda_t[t2] - self.lambda_t[t1]
        x0_combined = x0
        if t1_x0 is not None and h_prev is not None:
            r = h_prev / h
            x0_combined = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * t1_x0
        sigma_ratio = self.sqrt_one_sub_alpha_bar[t2] / self.sqrt_one_sub_alpha_bar[t1]
        # -expm1(-h) == 1 - e^{-h}
        x_t2 = sigma_ratio * xt + self.sqrt_alpha_bar[t2] * (-torch.expm1(-h)) * x0_combined
        return x_t2, x0, h

    @torch.no_grad()
    def __sample_dpm_2m(self, batch_size, image_shape, x=None, condition1=None, condition2=None, step=5, **kwargs):
        steps_arr = self.__get_steps_arr(step)
        if x is None:
            x = torch.randn(size=(batch_size, *image_shape), device=self.device)
        t1_x0, h_prev = None, None
        for i in tqdm(range(len(steps_arr) - 1), desc="DPM++ 2M Sampling"):
            x, t1_x0, h_prev = self.__clean_noise_dpm_2m(
                xt=x, t1=steps_arr[i], t2=steps_arr[i + 1], t1_x0=t1_x0, h_prev=h_prev,
                condition1=condition1, condition2=condition2, **kwargs)
        return x

    @torch.no_grad()
    def sample(self, batch_size, image_shape, condition1=None, condition2=None, mode="ddpm", x=None, step=5, sigma=0.,
               eta=None, **kwargs):
        assert mode in ["ddpm", "ddim", "dpm"], "支持 ddpm/ddim/dpm 采样"
        if mode == "ddpm":
            return self.__sample_ddpm(batch_size=batch_size, image_shape=image_shape, condition1=condition1,
                                      condition2=condition2, **kwargs)
        elif mode == "ddim":
            return self.__sample_ddim(batch_size=batch_size, image_shape=image_shape, x=x, condition1=condition1,
                                      condition2=condition2, step=step, sigma=sigma, eta=eta, **kwargs)
        else:
            return self.__sample_dpm_2m(batch_size=batch_size, image_shape=image_shape, x=x, condition1=condition1,
                                        condition2=condition2, step=step, **kwargs)

