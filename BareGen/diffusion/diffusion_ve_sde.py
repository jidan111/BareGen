from ..base_structs import *
from .timestep import *


class VEDiffusion(ConfigModule):
    """
    VE-SDE (Variance Exploding SDE): x_t = x_0 + sigma(t) * eps
    与 Diffusion(VP-SDE / DDPM) 的本质区别:
      1. 前向 SDE 没有漂移项 (f=0), 均值不衰减, 只有噪声方差 sigma^2 单调膨胀
      2. 终态是 N(0, sigma_max^2 I) 而不是 N(0, I), 所以采样起点必须乘 sigma_max
      3. 时间由 sigma 参数化 (log-sigma 或 rho 等间距), 而不是 alpha_bar / beta
      4. 网络默认套 EDM 预条件 (c_in/c_out/c_skip/c_noise), 否则 x_t 的幅度跨度可达 1e4, 很难训
    采样器:
      euler     概率流 ODE 一阶 (EDM Alg.1), 与 VE 下的 "DDIM" 等价, 也是默认
      heun      概率流 ODE 二阶 (EDM 默认), 一步两次前向
      dpm       DPM-Solver++(2M), 在 lambda = -log(sigma) 空间上做多步外推
      ancestral 逆向 SDE (Euler-Maruyama + 注入噪声), 随机采样
    """

    def __init__(self, model: ConfigModule, step_nums=100, step_dim=128,
                 schedule_name="karras", sigma_min=0.002, sigma_max=80.0, rho=7.0,
                 prediction_type="eps", loss_weighting="min_snr", min_snr_gamma=5.0,
                 preconditioning=True, time_cond="log_sigma", sigma_data=1.0,
                 train_sigma_sampling="uniform", p_mean=-1.2, p_std=1.2,
                 frequency_dim=256, max_period=10000.0, nominal_steps=1000.0):
        super(VEDiffusion, self).__init__()
        assert prediction_type in ["eps", "x0"], \
            f"prediction_type只支持 eps/x0, 但传入 {prediction_type}"
        assert loss_weighting in [None, "min_snr", "inverse_snr", "edm"], \
            f"loss_weighting只支持 None/min_snr/inverse_snr/edm, 但传入 {loss_weighting}"
        assert schedule_name in ["exp", "karras", "linear"], \
            f"schedule_name只支持 exp/karras/linear, 但传入 {schedule_name}"
        assert train_sigma_sampling in ["uniform", "log_uniform", "log_normal"], \
            f"train_sigma_sampling只支持 uniform/log_uniform/log_normal, 但传入 {train_sigma_sampling}"
        assert time_cond in ["log_sigma", "t"], \
            f"time_cond只支持 log_sigma/t, 但传入 {time_cond}"
        assert sigma_min > 0 and sigma_max > sigma_min, \
            f"需要 0 < sigma_min < sigma_max, 但传入 sigma_min={sigma_min}, sigma_max={sigma_max}"
        self.step_nums = step_nums
        self.prediction_type = prediction_type
        self.loss_weighting = loss_weighting
        self.min_snr_gamma = float(min_snr_gamma)
        self.preconditioning = preconditioning
        self.time_cond = time_cond
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.schedule_name = schedule_name
        self.train_sigma_sampling = train_sigma_sampling
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        # 连续时间步嵌入: 输入是 c_noise=log(sigma)/4 或归一化时间 t∈[0,1]
        self.step_embedding = TimestepEmbedding(dim=step_dim, step_nums=None,
                                                frequency_dim=frequency_dim, max_period=max_period,
                                                nominal_steps=nominal_steps)
        self.model = model
        self.config["model"] = self.model.config
        # reduction="none" 是为了按样本做 SNR / EDM 加权
        self.loss_func = nn.MSELoss(reduction="none")
        # 调度表完全由 schedule_name/sigma_min/sigma_max/rho 推导, 设成非持久化缓冲:
        # 不写进 checkpoint, 改 step_nums 后能自动重建, 也不会因形状变化加载失败
        for k, v in self.set_params().items():
            self.register_buffer(k, v, persistent=False)

    @property
    def device(self):
        """设备由参数实际所在位置推断, 不在 __init__ 里写死"""
        for param in self.parameters():
            return param.device
        for buffer in self.buffers():
            return buffer.device
        return torch.device("cpu")

    # ------------------------------ 调度表 ------------------------------

    def t_to_sigma(self, t):
        """连续时间 t∈[0,1] -> sigma (t=0 最干净, t=1 纯噪声)"""
        if self.schedule_name == "exp":
            # Song et al. 2021 原始 VE: sigma = sigma_min * (sigma_max/sigma_min)^t
            return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        if self.schedule_name == "karras":
            # EDM 的 rho 等间距: 在 sigma^(1/rho) 空间线性插值, 采样/训练步距更均匀
            inv_rho = 1.0 / self.rho
            return ((1 - t) * self.sigma_min ** inv_rho + t * self.sigma_max ** inv_rho) ** self.rho
        # linear
        return self.sigma_min + t * (self.sigma_max - self.sigma_min)

    def sigma_to_t(self, sigma):
        """sigma -> 连续时间 t∈[0,1], t_to_sigma 的逆函数"""
        sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max)
        if self.schedule_name == "exp":
            return torch.log(sigma / self.sigma_min) / math.log(self.sigma_max / self.sigma_min)
        if self.schedule_name == "karras":
            inv_rho = 1.0 / self.rho
            return (sigma ** inv_rho - self.sigma_min ** inv_rho) \
                / (self.sigma_max ** inv_rho - self.sigma_min ** inv_rho)
        return (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)

    def get_sigma_table(self):
        """长度为 step_nums 的 sigma 表, 索引 0 -> sigma_min(最干净), step_nums-1 -> sigma_max(纯噪声)"""
        t = torch.linspace(0.0, 1.0, steps=self.step_nums)
        sigma = self.t_to_sigma(t)
        return sigma.reshape(self.step_nums, 1, 1, 1)

    def set_params(self):
        sigma_table = self.get_sigma_table()
        return {"sigma_table": sigma_table}

    def get_sampling_sigmas(self, step_nums=None, device=None):
        """
        采样用的降序 sigma 序列, 长度 step_nums+1, 最后一个元素恒为 0
        末位补 0 是为了让最后一步 euler/heun 直接退化为输出 x0, 不需要额外特判
        """
        n = max(int(step_nums) if step_nums else self.step_nums, 1)
        device = device if device is not None else self.device
        t = torch.linspace(0.0, 1.0, steps=n, device=device, dtype=torch.float32)
        sigmas = torch.flip(self.t_to_sigma(t).reshape(-1), dims=[0])  # sigma_max -> sigma_min
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=torch.float32)], dim=0)
        return sigmas

    # ------------------------------ 前向加噪 ------------------------------

    def add_noise(self, x0, noise, t):
        """t 为离散步索引, VE 下均值不变只加噪"""
        return x0 + self.sigma_table[t].to(x0.dtype) * noise

    def add_noise_by_sigma(self, x0, noise, sigma):
        """按连续 sigma 加噪, sigma 可以是 [B] 或 [B,1,1,1]"""
        batch_size = x0.shape[0]
        sigma = sigma.reshape(batch_size, *([1] * (x0.dim() - 1))).to(x0.dtype)
        return x0 + sigma * noise

    def sample_train_sigma(self, batch_size, device=None):
        """训练时的 sigma 采样策略"""
        device = device if device is not None else self.device
        if self.train_sigma_sampling == "uniform":
            # 与 Diffusion 一致: 在离散表上均匀取步 (配合 karras 表等价于 rho 等间距)
            index = torch.randint(low=0, high=self.step_nums, size=(batch_size,), dtype=torch.long, device=device)
            return self.sigma_table[index].reshape(batch_size).to(device)
        if self.train_sigma_sampling == "log_uniform":
            u = torch.rand(size=(batch_size,), device=device)
            return torch.exp(math.log(self.sigma_min) + u * (math.log(self.sigma_max) - math.log(self.sigma_min)))
        # log_normal: EDM 默认, sigma = exp(N(p_mean, p_std))
        rnd = torch.randn(size=(batch_size,), device=device) * self.p_std + self.p_mean
        return torch.exp(rnd).clamp(min=self.sigma_min, max=self.sigma_max)

    # ------------------------------ 网络与预条件 ------------------------------

    def _precond_coef(self, sigma):
        """EDM Table 1 的预条件系数"""
        fp32 = sigma.to(torch.float32)
        sigma_data_2 = self.sigma_data ** 2
        c_skip = sigma_data_2 / (fp32 ** 2 + sigma_data_2)
        c_out = fp32 * self.sigma_data / torch.sqrt(fp32 ** 2 + sigma_data_2)
        c_in = 1.0 / torch.sqrt(fp32 ** 2 + sigma_data_2)
        c_noise = 0.25 * torch.log(fp32)
        return c_skip, c_out, c_in, c_noise

    def t_embed(self, t):
        """t: [B] float, 取值 [0,1] 或 c_noise 量纲"""
        return self.step_embedding(t)

    def denoise(self, xt, sigma, condition1=None, condition2=None, **kwargs):
        """
        去噪器 D(x; sigma) -> x0
        :param sigma: [B] 连续噪声强度
        """
        batch_size = xt.shape[0]
        sigma = sigma.reshape(batch_size).to(torch.float32)
        shape = (batch_size,) + (1,) * (xt.dim() - 1)
        c_skip, c_out, c_in, c_noise = self._precond_coef(sigma)
        if self.preconditioning:
            x_in = xt * c_in.reshape(shape).to(xt.dtype)
            # 开预条件时网络学到的是 EDM 的 F, prediction_type 只影响 loss 的计算口径
            t_in = c_noise if self.time_cond == "log_sigma" else self.sigma_to_t(sigma)
        else:
            x_in = xt
            t_in = self.sigma_to_t(sigma)
        t_embed = self.t_embed(t_in)
        out = self.model(x_in, t_embed, condition1, condition2, **kwargs)
        if self.preconditioning:
            return c_skip.reshape(shape).to(xt.dtype) * xt + c_out.reshape(shape).to(xt.dtype) * out
        if self.prediction_type == "eps":
            return xt - sigma.reshape(shape).to(xt.dtype) * out
        return out

    def _model_predict(self, xt, sigma, condition1=None, condition2=None, **kwargs):
        """统一入口: 去噪 -> (eps, x0), 其中 eps = (x_t - x_0) / sigma"""
        batch_size = xt.shape[0]
        shape = (batch_size,) + (1,) * (xt.dim() - 1)
        x0 = self.denoise(xt, sigma, condition1, condition2, **kwargs)
        eps = (xt - x0) / sigma.reshape(shape).to(xt.dtype).clamp(min=1e-12)
        return eps, x0

    # ------------------------------ 训练 ------------------------------

    def _loss_weight(self, sigma):
        """按噪声强度对 loss 加权, 返回 [B]"""
        fp32 = sigma.to(torch.float32)
        snr = 1.0 / torch.clamp(fp32 ** 2, min=1e-12)
        if self.loss_weighting == "min_snr":
            # Min-SNR-γ: min(SNR, γ) / SNR = min(1, γ·σ²)
            return torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma)) / snr
        if self.loss_weighting == "inverse_snr":
            return 1.0 / snr
        if self.loss_weighting == "edm":
            # EDM 权重 λ=(σ²+σ_d²)/(σ·σ_d)²; eps 参数化下要额外乘 σ²
            sigma_data_2 = self.sigma_data ** 2
            weight = (fp32 ** 2 + sigma_data_2) / sigma_data_2
            if self.prediction_type == "x0":
                weight = weight / torch.clamp(fp32 ** 2, min=1e-12)
            return weight
        return torch.ones_like(snr)

    def forward(self, x, condition1=None, condition2=None, **kwargs):
        batch_size, *_ = x.shape
        noise = torch.randn_like(x)
        sigma = self.sample_train_sigma(batch_size=batch_size, device=x.device)
        xt = self.add_noise_by_sigma(x0=x, noise=noise, sigma=sigma)
        eps, x0 = self._model_predict(xt, sigma, condition1, condition2, **kwargs)
        target = noise if self.prediction_type == "eps" else x
        prediction = eps if self.prediction_type == "eps" else x0
        loss = self.loss_func(prediction, target)
        loss = loss.reshape(loss.shape[0], -1).mean(dim=-1)
        if self.loss_weighting is not None:
            loss = loss * self._loss_weight(sigma).to(loss.dtype)
        return loss.mean()

    def loss(self, x, condition1=None, condition2=None, **kwargs):
        loss = self(x, condition1, condition2, **kwargs)
        return loss

    # ------------------------------ 单步去噪 ------------------------------

    @torch.no_grad()
    def __clean_noise_euler(self, xt, sigma_cur, sigma_next, condition1=None, condition2=None, **kwargs):
        """
        概率流 ODE 一阶 (EDM Alg.1): dx/dσ = (x - D(x;σ))/σ
        sigma_next=0 时结果正好等于 D(x;σ), 天然收尾到 x0
        """
        batch_size = xt.shape[0]
        sigma = torch.full((batch_size,), float(sigma_cur), device=xt.device, dtype=torch.float32)
        x0 = self.denoise(xt, sigma, condition1, condition2, **kwargs)
        d = (xt - x0) / max(float(sigma_cur), 1e-12)
        return xt + (float(sigma_next) - float(sigma_cur)) * d

    @torch.no_grad()
    def __clean_noise_heun(self, xt, sigma_cur, sigma_next, condition1=None, condition2=None, **kwargs):
        """
        概率流 ODE 二阶 (EDM Heun): 先用 euler 试探一步, 再用新旧两点斜率取平均
        最后一步 (sigma_next=0) 只做一阶, 避免除零
        """
        batch_size = xt.shape[0]
        sigma = torch.full((batch_size,), float(sigma_cur), device=xt.device, dtype=torch.float32)
        x0 = self.denoise(xt, sigma, condition1, condition2, **kwargs)
        d = (xt - x0) / max(float(sigma_cur), 1e-12)
        x_next = xt + (float(sigma_next) - float(sigma_cur)) * d
        if float(sigma_next) > 0:
            sigma_n = torch.full((batch_size,), float(sigma_next), device=xt.device, dtype=torch.float32)
            x0_next = self.denoise(x_next, sigma_n, condition1, condition2, **kwargs)
            d_prime = (x_next - x0_next) / float(sigma_next)
            x_next = xt + (float(sigma_next) - float(sigma_cur)) * (d + d_prime) / 2
        return x_next

    @torch.no_grad()
    def __clean_noise_ancestral(self, xt, sigma_cur, sigma_next, s_noise=1.0,
                                condition1=None, condition2=None, **kwargs):
        """
        逆向 VE-SDE (Euler-Maruyama):
            x_next = x + (σ_cur²-σ_next²)/σ_cur²·(D - x) + s_noise·√(σ_cur²-σ_next²)·z
        s_noise=0 时退化为确定性的精确解 (与 euler 一致)
        """
        batch_size = xt.shape[0]
        sigma = torch.full((batch_size,), float(sigma_cur), device=xt.device, dtype=torch.float32)
        x0 = self.denoise(xt, sigma, condition1, condition2, **kwargs)
        delta = max(float(sigma_cur) ** 2 - float(sigma_next) ** 2, 0.)
        x_next = xt + (delta / max(float(sigma_cur) ** 2, 1e-12)) * (x0 - xt)
        if s_noise > 0 and delta > 0:
            x_next = x_next + float(s_noise) * math.sqrt(delta) * torch.randn_like(xt)
        return x_next

    @torch.no_grad()
    def __clean_noise_dpm_2m(self, xt, sigma_cur, sigma_next, t1_x0=None, h_prev=None,
                             condition1=None, condition2=None, **kwargs):
        """
        DPM-Solver++(2M), VE 下 α≡1, λ = log(α/σ) = -log σ:
            h     = λ_next - λ_cur = log(σ_cur/σ_next)
            D̃     = (1 + 1/(2r))·D_i - (1/(2r))·D_{i-1},  r = h_prev/h
            x_next = (σ_next/σ_cur)·x + (1 - σ_next/σ_cur)·D̃
        :return: (x_next, x0@σ_cur, h)
        """
        batch_size = xt.shape[0]
        sigma = torch.full((batch_size,), float(sigma_cur), device=xt.device, dtype=torch.float32)
        x0 = self.denoise(xt, sigma, condition1, condition2, **kwargs)
        if float(sigma_next) <= 0:
            return x0, x0, None
        h = math.log(float(sigma_cur)) - math.log(float(sigma_next))
        x0_combined = x0
        if t1_x0 is not None and h_prev is not None and h > 0:
            r = h_prev / h
            x0_combined = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * t1_x0
        ratio = float(sigma_next) / float(sigma_cur)
        return ratio * xt + (1 - ratio) * x0_combined, x0, h

    # ------------------------------ 采样循环 ------------------------------

    @torch.no_grad()
    def sample_euler(self, x, step_nums=None, condition1=None, condition2=None, callback=None, **kwargs):
        sigmas = self.get_sampling_sigmas(step_nums=step_nums, device=x.device)
        for i in tqdm(range(len(sigmas) - 1), desc="VE Euler Sampling"):
            x = self.__clean_noise_euler(x, sigmas[i], sigmas[i + 1],
                                         condition1=condition1, condition2=condition2, **kwargs)
            if callback is not None:
                callback(x, i)
        return x

    @torch.no_grad()
    def sample_heun(self, x, step_nums=None, condition1=None, condition2=None, callback=None, **kwargs):
        sigmas = self.get_sampling_sigmas(step_nums=step_nums, device=x.device)
        for i in tqdm(range(len(sigmas) - 1), desc="VE Heun Sampling"):
            x = self.__clean_noise_heun(x, sigmas[i], sigmas[i + 1],
                                        condition1=condition1, condition2=condition2, **kwargs)
            if callback is not None:
                callback(x, i)
        return x

    @torch.no_grad()
    def sample_ancestral(self, x, step_nums=None, s_noise=1.0, condition1=None, condition2=None, callback=None,
                         **kwargs):
        sigmas = self.get_sampling_sigmas(step_nums=step_nums, device=x.device)
        for i in tqdm(range(len(sigmas) - 1), desc="VE SDE Sampling"):
            x = self.__clean_noise_ancestral(x, sigmas[i], sigmas[i + 1], s_noise=s_noise,
                                             condition1=condition1, condition2=condition2, **kwargs)
            if callback is not None:
                callback(x, i)
        return x

    @torch.no_grad()
    def sample_dpm_2m(self, x, step_nums=None, condition1=None, condition2=None, callback=None, **kwargs):
        sigmas = self.get_sampling_sigmas(step_nums=step_nums, device=x.device)
        t1_x0, h_prev = None, None
        for i in tqdm(range(len(sigmas) - 1), desc="VE DPM++ 2M Sampling"):
            x, t1_x0, h_prev = self.__clean_noise_dpm_2m(
                xt=x, sigma_cur=sigmas[i], sigma_next=sigmas[i + 1], t1_x0=t1_x0, h_prev=h_prev,
                condition1=condition1, condition2=condition2, **kwargs)
            if callback is not None:
                callback(x, i)
        return x

    @torch.no_grad()
    def sample(self, batch_size, image_shape, condition1=None, condition2=None, mode="euler", x=None,
               step_nums=None, step=1, s_noise=1.0, callback=None, **kwargs):
        """
        :param step_nums: 采样步数, 默认 self.step_nums
        :param step: 兼容 Diffusion 的习惯, 在默认步数上抽稀 (step=5 表示只用 1/5 的步数); step_nums 优先
        :param s_noise: 仅 ancestral 生效, 随机项缩放
        :param callback: 可选, fn(x_t, step_index), 每一步去噪后调用一次, 用于抓取中间帧做 gif
        """
        assert mode in ["euler", "heun", "dpm", "ancestral", "ddim", "sde"], \
            f"支持 euler/heun/dpm/ancestral 采样, 但传入 {mode}"
        if step_nums is None:
            step_nums = max(int(self.step_nums / max(int(step), 1)), 1)
        if x is None:
            # VE 的终态是 N(0, sigma_max^2 I), 起点必须乘 sigma_max
            x = torch.randn(size=(batch_size, *image_shape), device=self.device) * self.sigma_max
        if mode in ["euler", "ddim"]:
            return self.sample_euler(x=x, step_nums=step_nums, condition1=condition1,
                                     condition2=condition2, callback=callback, **kwargs)
        if mode == "heun":
            return self.sample_heun(x=x, step_nums=step_nums, condition1=condition1,
                                    condition2=condition2, callback=callback, **kwargs)
        if mode in ["ancestral", "sde"]:
            return self.sample_ancestral(x=x, step_nums=step_nums, s_noise=s_noise,
                                         condition1=condition1, condition2=condition2, callback=callback, **kwargs)
        return self.sample_dpm_2m(x=x, step_nums=step_nums, condition1=condition1,
                                  condition2=condition2, callback=callback, **kwargs)

