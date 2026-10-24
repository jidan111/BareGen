from .import_packages import *


def get_all_file_paths(root_path, ends_with=("jpg", "jpeg", "png")) -> list:
    file_path_list = []
    # 遍历文件夹
    for dir_path, dir_names, file_names in os.walk(root_path):
        for file_name in file_names:
            if file_name.endswith(tuple(ends_with)):
                full_path = os.path.join(dir_path, file_name)
                file_path_list.append(full_path)
    return file_path_list


def tensor_to_pil(x, nrow=None, padding=1, value_range=(-1, 1)) -> Image.Image:
    """
    把采样过程中的张量转成一张 PIL 图, 供 GIF 使用
    :param x: [C,H,W] / [B,C,H,W] / [N,B,C,H,W] 任意浮点张量
    :param value_range: 输入张量的数值区间, 默认 (-1,1); 若已归一化到 [0,1] 传 (0,1)
    """
    if x.dim() == 5:  # [N,B,C,H,W] -> [N*B,C,H,W]
        x = x.flatten(start_dim=0, end_dim=1)
    if x.dim() == 3:
        x = x.unsqueeze(dim=0)
    x = x.detach().to(torch.float32).cpu()
    low, high = value_range
    x = (x - low) / max(high - low, 1e-12)
    nrow = nrow if nrow is not None else max(int(math.sqrt(x.shape[0])), 1)
    # padding 时 make_grid 用 0 填充, 背景会偏暗, 这里手动填白边由 pad_value 控制
    grid = make_grid(x, nrow=nrow, padding=padding, normalize=False, pad_value=1.0)
    ndarr = grid.clamp(min=0., max=1.).mul(255).add(0.5).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(ndarr)


def save_gif(frames, fp, fps=20, nrow=None, padding=1, value_range=(-1, 1), loop=0, optimize=True,
             duration_each=None):
    """
    把一段采样轨迹保存成 .gif 动态图
    :param frames: list[Tensor], 每个元素为 [B,C,H,W] / [C,H,W]; 或 [N,B,C,H,W] 的 5 维张量
    :param fp: 输出路径, 以 .gif 结尾
    :param fps: 帧率
    :param loop: 循环次数, 0 表示无限循环
    :param duration_each: 若传入, 覆盖 fps, 单位为毫秒
    """
    assert len(frames) > 0, "frames 为空, 没有可写入的帧"
    if torch.is_tensor(frames):
        frames = list(frames.unbind(dim=0))
    dir_name = os.path.dirname(fp)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    images = [tensor_to_pil(f, nrow=nrow, padding=padding, value_range=value_range) for f in frames]
    duration = int(duration_each) if duration_each is not None else max(int(1000 / max(fps, 1e-6)), 1)
    images[0].save(fp, format="GIF", save_all=True, append_images=images[1:],
                   duration=duration, loop=loop, optimize=optimize)
    return fp


def sample_trajectory(model, batch_size=None, image_shape=None, x=None, keep_every=1, keep_last=True,
                      to_cpu=True, **sample_kwargs):
    """
    通用轨迹采样: 在 sample() 的每一步回调里收集 x_t, 返回 [N,B,C,H,W]
    适用于 Diffusion(ddpm/ddim/dpm) / RectifiedFlow(euler/rk4) / VEDiffusion(euler/heun/dpm/ancestral)
    :param model: 采样器实例, 需要有 .sample(..., callback=...)
    :param keep_every: 每隔多少步保留一帧, 1 表示每步都留(帧数=步数), 调大可压缩 gif 体积
    :param keep_last: 是否强制保留最终帧
    """
    frames = []
    total = sample_kwargs.get("step_nums", None)

    def _callback(x_t, step_idx):
        if keep_every > 1 and step_idx % keep_every != 0:
            return
        frames.append(x_t.detach().to("cpu") if to_cpu else x_t.detach().clone())

    out = model.sample(batch_size=batch_size, image_shape=image_shape, x=x, callback=_callback, **sample_kwargs)
    if keep_last:
        last = out.detach().to("cpu") if to_cpu else out.detach().clone()
        if not frames or not torch.equal(frames[-1], last):
            frames.append(last)
    return torch.stack(frames, dim=0), out


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total:,}  ({total / 1e6:.2f} M)")
    print(f"可训练量: {trainable:,}  ({trainable / 1e6:.2f} M)")
    # 顺便按子模块拆开看
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:15s}: {n:>12,}  ({n / 1e6:.2f} M)")
