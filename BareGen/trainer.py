from .import_packages import *
from .base_structs import *
from .losses import *
from .loger import *


def resolve_amp_dtype(precision="auto"):
    if precision == "fp32":
        return None, False  # 注意: 不能把这个 None 直接喂给 autocast,
        #        autocast(dtype=None) 会回落到默认 fp16, 必须显式 enabled=False
    if precision == "bf16":
        return torch.bfloat16, False  # bf16 指数位和 fp32 一样, 不会溢出/下溢
    if precision == "fp16":
        return torch.float16, True  # fp16 最小正规数 6e-5, 不开 scaler 梯度会大量下溢
    if precision != "auto":
        raise ValueError(
            f"amp_precision 只支持 auto/fp32/bf16/fp16, 但传入 {precision!r}")
    # auto: Ampere+(sm_80) 有 bf16 tensor core, 又快又稳; 更早的架构只能走 fp16+scaler
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16, False
    return torch.float16, True


class Trainer(object):
    def __init__(self, middle_validate_step=None, save_path="./model", valid_path="./valid",
                 using_ema=False,
                 ema_update_step=10,
                 ema_decay=0.999, amp_precision="auto"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.middle_validate_flag = middle_validate_step is not None
        self.middle_validate_step = middle_validate_step
        self.save_path = save_path
        self.valid_path = valid_path
        self.log_path = os.path.join(self.save_path, "log")
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.valid_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        self.using_ema = using_ema
        self.ema_update_step = ema_update_step
        self.ema = EMA(decay=ema_decay)
        self.training_dtype, self.using_scaler_flag = resolve_amp_dtype(precision=amp_precision)
        self.amp_precision = amp_precision
        # dtype 为 None 意味着纯 fp32, 必须显式关掉 autocast,
        # 否则 autocast(dtype=None) 会按 CUDA 默认 dtype(fp16) 生效
        self.amp_enabled = self.training_dtype is not None

    def amp_autocast(self):
        return autocast(device_type=self.device, dtype=self.training_dtype, enabled=self.amp_enabled)

    def set_model(self, *args, **kwargs):
        raise NotImplementedError

    def set_optimizer(self, *args, **kwargs):
        raise NotImplementedError

    def validate_loss(self, loss, optimizer):
        if not torch.isfinite(loss).all():
            optimizer.zero_grad()
            raise Exception("训练出现空值，已终止训练")

    def validate(self, epoch, **kwargs) -> None:
        raise NotImplementedError

    def save_checkpoint(self) -> None:
        raise NotImplementedError

    def preprocessing_data(self, data):
        raise NotImplementedError

    def train_one_batch(self, data, iter_cnt) -> dict:
        raise NotImplementedError

    def train_one_epoch(self, dataloader, epoch_cnt, epoch, log_step, **kwargs):
        loger = WriterLoger(loger_dir=self.log_path, loger_name=f"log_{epoch_cnt}.jsonl")
        for cnt, data in enumerate(tqdm(dataloader, desc=f"{epoch_cnt}/{epoch}")):
            data = self.preprocessing_data(data)
            batch_loss = self.train_one_batch(data=data, iter_cnt=cnt)
            if cnt % log_step == 0:
                loger(item=batch_loss)
            if self.middle_validate_flag:
                if cnt % self.middle_validate_step == 0:
                    self.validate(epoch=f"{epoch}_{epoch_cnt}_{cnt}", **kwargs)
                    self.save_checkpoint()

    def run(self, dataloader, epoch=100, valid_step=1, log_step=10, **kwargs):
        for epoch_cnt in range(epoch):
            self.train_one_epoch(dataloader=dataloader, epoch_cnt=epoch_cnt, epoch=epoch, log_step=log_step, **kwargs)
            if epoch_cnt % valid_step == 0:
                self.validate(epoch=epoch_cnt, **kwargs)
                self.save_checkpoint()


class BaseTrainer(Trainer):
    def __init__(self, middle_validate_step=None,
                 gradient_accumulation_step=1,
                 using_ema=False,
                 ema_update_step=10,
                 ema_decay=0.999,
                 save_path="./model",
                 valid_path="./valid",
                 compile_model=False, amp_precision="auto"):
        super(BaseTrainer, self).__init__(middle_validate_step=middle_validate_step, save_path=save_path,
                                          valid_path=valid_path, ema_update_step=ema_update_step, using_ema=using_ema,
                                          ema_decay=ema_decay, amp_precision=amp_precision)
        self.gradient_accumulation_step = gradient_accumulation_step
        self.gradient_accumulation_step_cur = 0
        self.using_ema = using_ema
        self.compile_model = compile_model
        self.model = ConfigModule()
        self.optimizer = torch.optim.Optimizer
        self.scheduler = torch.optim.lr_scheduler.Optimizer
        self.scale = GradScaler(device=self.device, enabled=self.using_scaler_flag)

    def set_model(self, model: ConfigModule):
        if not self.compile_model:
            self.model = model.to(self.device)
        else:
            self.model = torch.compile(model).to(self.device)
        if self.using_ema:
            self.ema.set_shadow(self.model)

    def set_optimizer(self, lr=1e-4, total_train_steps=3_600_000_000, warmup_steps=1):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                                           weight_decay=0.03)
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-8,  # 起始几乎0lr
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_steps = total_train_steps - warmup_steps
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_steps,
            eta_min=lr * 0.01
        )
        # 串联两个scheduler：先跑warmup，再跑cosine
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                      milestones=[warmup_steps])

    def preprocessing_data(self, data):
        data = data.to(self.device)
        return data

    def validate(self, epoch, **kwargs) -> None:
        pass

    def save_checkpoint(self) -> None:
        config = self.model.config
        if self.compile_model:
            state_dict = self.model._orig_mod.state_dict()
        else:
            state_dict = self.model.state_dict()
        model_name = list(config.keys())[0]
        config_path = os.path.join(self.save_path, f"{model_name}.json")
        state_dict_path = os.path.join(self.save_path, f"{model_name}.pth")
        with open(config_path, 'w') as f:
            f.write(json.dumps(config))
        torch.save(state_dict, state_dict_path)
        if self.using_ema:
            ema_path = os.path.join(self.save_path, f"ema_{model_name}.pth")
            torch.save(self.ema.shadow, ema_path)

    def compute_loss(self, data):
        loss = self.model.loss(data)
        self.validate_loss(loss=loss, optimizer=self.optimizer)
        return loss

    def train_one_batch(self, data, iter_cnt) -> dict:
        self.model.train()
        with self.amp_autocast():
            loss = self.compute_loss(data) / self.gradient_accumulation_step
        self.scale.scale(loss).backward()
        self.gradient_accumulation_step_cur += 1
        if self.gradient_accumulation_step_cur == self.gradient_accumulation_step:
            self.scale.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scale.step(self.optimizer)
            self.scale.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.gradient_accumulation_step_cur = 0
        if self.using_ema:
            if iter_cnt % self.ema_update_step == 0:
                self.ema.update(self.model)
        return {"loss": loss.item()}


class BaseGanTrainer(Trainer):
    def __init__(self, generator_train_step, loss_func=hinge, middle_validate_step=None,
                 using_ema=False, ema_update_step=10, ema_decay=0.999,
                 save_path="./model", valid_path="./valid", amp_precision="auto"):
        super().__init__(middle_validate_step=middle_validate_step, save_path=save_path,
                         valid_path=valid_path, ema_update_step=ema_update_step, using_ema=using_ema,
                         ema_decay=ema_decay, amp_precision=amp_precision)
        self.generator_train_step = generator_train_step
        self.loss_func = loss_func
        self.generator_train_step_cur = 0
        self.generator = ConfigModule()
        self.generator_optimizer = torch.optim.Optimizer
        self.generator_scheduler = torch.optim.lr_scheduler.Optimizer
        self.generator_scale = GradScaler(device=self.device, enabled=self.using_scaler_flag)
        self.discriminator = ConfigModule()
        self.discriminator_optimizer = torch.optim.Optimizer
        self.discriminator_scheduler = torch.optim.lr_scheduler.Optimizer
        self.discriminator_scale = GradScaler(device=self.device, enabled=self.using_scaler_flag)

    def set_model(self, generator, discriminator):
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        if self.using_ema:
            self.ema.set_shadow(self.generator)

    def set_optimizer(self, generator_lr, discriminator_lr, total_train_steps=3_600_000, warmup_steps=1):
        def func(lr, model):
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                                          weight_decay=0.03)
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-7,  # 起始几乎0lr
                end_factor=1.0,
                total_iters=warmup_steps
            )
            cosine_steps = total_train_steps - warmup_steps
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=lr * 0.01
            )
            # 串联两个scheduler：先跑warmup，再跑cosine
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                     milestones=[warmup_steps])
            return optimizer, scheduler

        self.generator_optimizer, self.generator_scheduler = func(generator_lr, self.generator)
        self.discriminator_optimizer, self.discriminator_scheduler = func(discriminator_lr, self.discriminator)

    def validate(self, epoch, **kwargs) -> None:
        file_path = os.path.join(self.valid_path, f"{epoch}.jpg")
        valid_batch_size = kwargs.get("valid_batch_size", 4)
        row = int(math.sqrt(valid_batch_size))
        self.generator.eval()
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.generator)
        with torch.no_grad():
            in_dim = self.generator.in_dim
            noise = torch.randn(size=(valid_batch_size, in_dim), device=self.device)
            sample = self.generator(noise).clamp(min=-1, max=1)
            save_image((sample.to(torch.float32) + 1) / 2, fp=file_path, nrow=row,
                       normalize=False, padding=1)
        if self.using_ema:
            self.ema.restore(model=self.generator, backup=back_params)
        self.generator.train()

    def save_checkpoint(self) -> None:
        def func(model):
            config = model.config
            state_dict = model.state_dict()
            model_name = list(config.keys())[0]
            config_path = os.path.join(self.save_path, f"{model_name}.json")
            state_dict_path = os.path.join(self.save_path, f"{model_name}.pth")
            with open(config_path, 'w') as f:
                f.write(json.dumps(config))
            torch.save(state_dict, state_dict_path)
            return model_name

        generator_name = func(self.generator)
        discriminator_name = func(self.discriminator)
        if self.using_ema:
            ema_path = os.path.join(self.save_path, f"ema_{generator_name}.pth")
            torch.save(self.ema.shadow, ema_path)

    def preprocessing_data(self, data):
        data = data.to(self.device)
        return data

    def train_generator_once(self, data) -> float:
        self.generator.train()
        self.generator_optimizer.zero_grad()
        with self.amp_autocast():
            inputs = self.generator(data)
            loss = -self.discriminator(inputs).mean()
        self.validate_loss(loss=loss, optimizer=self.generator_optimizer)
        self.generator_scale.scale(loss).backward()
        self.generator_scale.unscale_(self.generator_optimizer)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.generator_scale.step(self.generator_optimizer)
        self.generator_scale.update()
        self.generator_scheduler.step()
        return loss.item()

    def train_discriminator_once(self, inputs, targets) -> float:
        self.discriminator.train()
        self.discriminator_optimizer.zero_grad()
        with self.amp_autocast():
            inputs = self.discriminator(inputs.detach())
            targets = self.discriminator(targets)
            loss = self.loss_func(inputs, targets)
        self.validate_loss(loss=loss, optimizer=self.discriminator_optimizer)
        self.discriminator_scale.scale(loss).backward()
        self.discriminator_scale.unscale_(self.discriminator_optimizer)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.discriminator_scale.step(self.discriminator_optimizer)
        self.discriminator_scale.update()
        self.discriminator_scheduler.step()
        return loss.item()

    def train_one_batch(self, data, iter_cnt) -> dict:
        self.generator_train_step_cur += 1
        gen_loss = 0.
        noise = torch.randn(size=(data.shape[0], self.generator.in_dim), device=self.device)
        inputs = self.generator(noise)
        dis_loss = self.train_discriminator_once(inputs=inputs, targets=data)
        if self.generator_train_step_cur == self.generator_train_step:
            noise = torch.randn(size=(data.shape[0], self.generator.in_dim), device=self.device)
            gen_loss = self.train_generator_once(data=noise)
            self.generator_train_step_cur = 0
        if self.using_ema:
            if iter_cnt % self.ema_update_step == 0:
                self.ema.update(self.generator)
        return {"g_loss": gen_loss, "d_loss": dis_loss}


class CLIPTrainer(BaseTrainer):
    def preprocessing_data(self, data):
        image = data[0].to(self.device)
        token = data[1].to(self.device)
        return image, token

    def compute_loss(self, data):
        loss, image_loss, text_loss = self.model.loss(data[0], data[1])
        self.validate_loss(loss=loss, optimizer=self.optimizer)
        return loss, image_loss, text_loss

    def train_one_batch(self, data, iter_cnt) -> dict:
        self.model.train()
        with self.amp_autocast():
            loss, image_loss, text_loss = self.compute_loss(data)
            loss = loss / self.gradient_accumulation_step
        self.scale.scale(loss).backward()
        self.gradient_accumulation_step_cur += 1
        if self.gradient_accumulation_step_cur == self.gradient_accumulation_step:
            self.scale.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scale.step(self.optimizer)
            self.scale.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.gradient_accumulation_step_cur = 0
        if self.using_ema:
            if iter_cnt % self.ema_update_step == 0:
                self.ema.update(self.model)
        return {"loss": loss.item(), "image_loss": image_loss.item(), "text_loss": text_loss.item()}

    def validate(self, epoch, **kwargs) -> None:
        assert "valid_data" in list(kwargs.keys()), "必须显示传入valid_data"
        valid_data = self.preprocessing_data(kwargs["valid_data"])
        valid_image = valid_data[0]
        valid_token = valid_data[1]
        self.model.eval()
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.model)
        with torch.no_grad():
            recall_top_k = self.model.recall_at_k(valid_image, valid_token, top_k_arr=(1, 5, 10))
        if self.using_ema:
            self.ema.restore(model=self.model, backup=back_params)
        self.model.train()
        return recall_top_k


class DiffusionTrainer(BaseTrainer):
    def validate(self, epoch, **kwargs) -> None:
        assert "image_shape" in list(kwargs.keys()), "必须显式传入image_shape参数"
        image_shape = kwargs["image_shape"]
        file_path = os.path.join(self.valid_path, f"{epoch}.jpg")
        valid_batch_size = kwargs.get("valid_batch_size", 4)
        row = int(math.sqrt(valid_batch_size))
        if self.model.__class__.__name__ == "Diffusion":
            mode = kwargs.get("mode", "dpm")
        else:
            mode = kwargs.get("mode", "euler")
        self.model.eval()
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.model)
        with torch.no_grad():
            sample = self.model.sample(batch_size=valid_batch_size, mode=mode, image_shape=image_shape).clamp(min=-1,
                                                                                                              max=1)
            save_image((sample.to(torch.float32) + 1) / 2, fp=file_path, nrow=row,
                       normalize=False, padding=1)
        if self.using_ema:
            self.ema.restore(model=self.model, backup=back_params)
        self.model.train()


class ESRTrainer(BaseTrainer):
    def preprocessing_data(self, data):
        lr = data[0].to(self.device)
        hr = data[1].to(self.device)
        return lr, hr

    def compute_loss(self, data):
        loss = self.model.loss(data[0], data[1])
        self.validate_loss(loss=loss, optimizer=self.optimizer)
        return loss

    def validate(self, epoch, **kwargs) -> None:
        assert "valid_data" in list(kwargs.keys()), "必须显示传入valid_data"
        valid_data = kwargs["valid_data"].to(self.device)
        file_path = os.path.join(self.valid_path, f"{epoch}.jpg")
        valid_batch_size = valid_data.shape[0]
        row = int(math.sqrt(valid_batch_size))
        self.model.eval()
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.model)
        with torch.no_grad():
            out = self.model(valid_data).clamp(min=-1, max=1)
            out_shape = (out.shape[2], out.shape[3], out.shape[1])
            in_shape = (valid_data.shape[2], valid_data.shape[3], valid_data.shape[1])
            outs = make_grid(out.to(torch.float32), nrow=row, normalize=True,
                             padding=1).detach().cpu().numpy().transpose(1, 2, 0).astype(
                np.float32)
            inputs = make_grid(valid_data.to(torch.float32), nrow=row, normalize=True,
                               padding=1).detach().cpu().numpy().transpose(1, 2,
                                                                           0).astype(
                np.float32)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
        ax1.imshow(inputs, cmap='gray' if in_shape[-1] == 1 else None)
        ax1.axis('off')
        ax1.set_title(f'Input (Shape: {in_shape})', fontsize=12, pad=15)
        ax2.imshow(outs, cmap='gray' if out_shape[-1] == 1 else None)
        ax2.axis('off')
        ax2.set_title(f'Output (Shape: {out_shape})', fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.close()
        if self.using_ema:
            self.ema.restore(model=self.model, backup=back_params)
        self.model.train()


class AutoEncoderTrainer(BaseTrainer):
    def validate(self, epoch, **kwargs) -> None:
        assert "valid_data" in list(kwargs.keys()), "必须显示传入valid_data"
        valid_data = self.preprocessing_data(kwargs["valid_data"])
        file_path = os.path.join(self.valid_path, f"{epoch}.jpg")
        valid_batch_size = valid_data.shape[0]
        row = int(math.sqrt(valid_batch_size))
        self.model.eval()
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.model)
        with torch.no_grad():
            out, latent = self.model(valid_data)
            out = out.clamp(min=-1, max=1)
            latent = latent.mode().clamp(min=-1, max=1)
            latent_shape = (latent.shape[2], latent.shape[3], latent.shape[1])
            latent = \
                make_grid(latent.to(torch.float32), nrow=row, normalize=True,
                          padding=1).detach().cpu().numpy().transpose(1,
                                                                      2,
                                                                      0).astype(
                    np.float32)[:, :, :3]
            out_shape = (out.shape[2], out.shape[3], out.shape[1])
            in_shape = (valid_data.shape[2], valid_data.shape[3], valid_data.shape[1])
            outs = make_grid(out.to(torch.float32), nrow=row, normalize=True,
                             padding=1).detach().cpu().numpy().transpose(1, 2, 0).astype(
                np.float32)
            inputs = make_grid(valid_data.to(torch.float32), nrow=row, normalize=True,
                               padding=1).detach().cpu().numpy().transpose(1, 2,
                                                                           0).astype(
                np.float32)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(inputs, cmap='gray' if in_shape[-1] == 1 else None)
        ax1.axis('off')
        ax1.set_title(f'Input (Shape: {in_shape})', fontsize=12, pad=15)
        ax2.imshow(latent, cmap='gray' if latent_shape[-1] == 1 else None)
        ax2.axis('off')
        ax2.set_title(f'Latent (Shape: {latent_shape})', fontsize=12, pad=15)
        ax3.imshow(outs, cmap='gray' if out_shape[-1] == 1 else None)
        ax3.axis('off')
        ax3.set_title(f'Output (Shape: {out_shape})', fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.close()
        if self.using_ema:
            self.ema.restore(model=self.model, backup=back_params)
        self.model.train()


class BaseTrainerWithDiscriminator(BaseGanTrainer):
    def __init__(self, dis_start, loss_func=hinge, middle_validate_step=None,
                 using_ema=False, ema_update_step=10, ema_decay=0.999,
                 save_path="./model", valid_path="./valid",amp_precision="auto"):
        super(BaseTrainerWithDiscriminator, self).__init__(generator_train_step=-1, loss_func=loss_func,
                                                           middle_validate_step=middle_validate_step,
                                                           using_ema=using_ema, ema_update_step=ema_update_step,
                                                           ema_decay=ema_decay,
                                                           save_path=save_path, valid_path=valid_path,amp_precision=amp_precision)
        self.dis_start = dis_start
        self.dis_start_cur = 0
        self.dis_train_flag = False

    def calculate_adaptive_weight(self, a_loss, b_loss, model_last_layer):
        a_grads = autograd.grad(outputs=a_loss, inputs=model_last_layer, retain_graph=True)[0]
        b_grads = autograd.grad(outputs=b_loss, inputs=model_last_layer, retain_graph=True)[0]
        b_weight = torch.norm(a_grads) / (torch.norm(b_grads) + 1e-4)
        b_weight = torch.clamp(b_weight, 0.0, 1e4).detach()
        return b_weight

    def generator_rec_loss(self, data):
        loss, out = self.generator.loss(data, return_features=True)
        return loss, out

    def train_generator_once(self, data):
        self.generator.train()
        self.generator_optimizer.zero_grad()
        with self.amp_autocast():
            rec_loss, out = self.generator_rec_loss(data)
            if self.dis_train_flag:
                g_loss = - self.discriminator(out).mean()
                g_weight = self.calculate_adaptive_weight(rec_loss, g_loss, self.generator.get_last_layer_weight())
                loss = rec_loss + g_weight * g_loss
            else:
                g_loss = torch.tensor(0.0)
                loss = rec_loss
        self.validate_loss(loss=loss, optimizer=self.generator_optimizer)
        self.generator_scale.scale(loss).backward()
        self.generator_scale.unscale_(self.generator_optimizer)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.generator_scale.step(self.generator_optimizer)
        self.generator_scale.update()
        self.generator_scheduler.step()
        self.dis_start_cur += 1
        self.dis_train_flag = True if self.dis_start_cur >= self.dis_start else False
        return rec_loss.item(), g_loss.item(), out

    def train_one_batch(self, data, iter_cnt):
        dis_loss = 0.
        rec_loss, g_loss, inputs = self.train_generator_once(data)
        if self.dis_train_flag:
            dis_loss = self.train_discriminator_once(inputs=inputs, targets=data)
        if self.using_ema:
            if iter_cnt % self.ema_update_step == 0:
                self.ema.update(self.generator)
        return {"rec_loss": rec_loss, "g_loss": g_loss, "d_loss": dis_loss}

    def validate(self, epoch, **kwargs) -> None:
        assert "valid_data" in list(kwargs.keys()), "必须显示传入valid_data"
        valid_data = self.preprocessing_data(kwargs["valid_data"])
        file_path = os.path.join(self.valid_path, f"{epoch}.jpg")
        valid_batch_size = valid_data.shape[0]
        row = int(math.sqrt(valid_batch_size))
        self.generator.eval()
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.generator)
        with torch.no_grad():
            out, latent = self.generator(valid_data)
            out = out.clamp(min=-1, max=1)
            latent = latent.mode().clamp(min=-1, max=1)
            latent_shape = (latent.shape[2], latent.shape[3], latent.shape[1])
            latent = \
                make_grid(latent.to(torch.float32), nrow=row, normalize=True,
                          padding=1).detach().cpu().numpy().transpose(1,
                                                                      2,
                                                                      0).astype(
                    np.float32)[:, :, :3]
            out_shape = (out.shape[2], out.shape[3], out.shape[1])
            in_shape = (valid_data.shape[2], valid_data.shape[3], valid_data.shape[1])
            outs = make_grid(out.to(torch.float32), nrow=row, normalize=True,
                             padding=1).detach().cpu().numpy().transpose(1, 2, 0).astype(
                np.float32)
            inputs = make_grid(valid_data.to(torch.float32), nrow=row, normalize=True,
                               padding=1).detach().cpu().numpy().transpose(1, 2,
                                                                           0).astype(
                np.float32)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(inputs, cmap='gray' if in_shape[-1] == 1 else None)
        ax1.axis('off')
        ax1.set_title(f'Input (Shape: {in_shape})', fontsize=12, pad=15)
        ax2.imshow(latent, cmap='gray' if latent_shape[-1] == 1 else None)
        ax2.axis('off')
        ax2.set_title(f'Latent (Shape: {latent_shape})', fontsize=12, pad=15)
        ax3.imshow(outs, cmap='gray' if out_shape[-1] == 1 else None)
        ax3.axis('off')
        ax3.set_title(f'Output (Shape: {out_shape})', fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.close()
        if self.using_ema:
            self.ema.restore(model=self.generator, backup=back_params)
        self.generator.train()


class ESRTrainerWithDiscriminator(BaseTrainerWithDiscriminator):
    def preprocessing_data(self, data):
        lr = data[0].to(self.device)
        hr = data[1].to(self.device)
        return lr, hr

    def generator_rec_loss(self, data):
        loss, out = self.generator.loss(data[0], data[1], return_features=True)
        return loss, out

    def validate(self, epoch, **kwargs) -> None:
        assert "valid_data" in list(kwargs.keys()), "必须显示传入valid_data"
        valid_data = kwargs["valid_data"].to(self.device)
        file_path = os.path.join(self.valid_path, f"{epoch}.jpg")
        valid_batch_size = valid_data.shape[0]
        row = int(math.sqrt(valid_batch_size))
        self.generator.eval()
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.generator)
        with torch.no_grad():
            out = self.generator(valid_data).clamp(min=-1, max=1)
            out_shape = (out.shape[2], out.shape[3], out.shape[1])
            in_shape = (valid_data.shape[2], valid_data.shape[3], valid_data.shape[1])
            outs = make_grid(out.to(torch.float32), nrow=row, normalize=True,
                             padding=1).detach().cpu().numpy().transpose(1, 2, 0).astype(
                np.float32)
            inputs = make_grid(valid_data.to(torch.float32), nrow=row, normalize=True,
                               padding=1).detach().cpu().numpy().transpose(1, 2,
                                                                           0).astype(
                np.float32)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
        ax1.imshow(inputs, cmap='gray' if in_shape[-1] == 1 else None)
        ax1.axis('off')
        ax1.set_title(f'Input (Shape: {in_shape})', fontsize=12, pad=15)
        ax2.imshow(outs, cmap='gray' if out_shape[-1] == 1 else None)
        ax2.axis('off')
        ax2.set_title(f'Output (Shape: {out_shape})', fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.close()
        if self.using_ema:
            self.ema.restore(model=self.generator, backup=back_params)
        self.generator.train()


class Text2ImageTrainer(BaseTrainer):
    def __init__(self, tokenizer, clip: ConfigModule, autoencoder: ConfigModule, latent_std: torch.Tensor,
                 latent_mean: torch.Tensor, middle_validate_step=None,
                 gradient_accumulation_step=1,
                 using_ema=False,
                 ema_update_step=10,
                 ema_decay=0.999,
                 save_path="./model",
                 valid_path="./valid",
                 compile_model=False, amp_precision="auto"):
        super(Text2ImageTrainer, self).__init__(middle_validate_step=middle_validate_step,
                                                gradient_accumulation_step=gradient_accumulation_step,
                                                using_ema=using_ema,
                                                ema_update_step=ema_update_step,
                                                ema_decay=ema_decay,
                                                save_path=save_path,
                                                valid_path=valid_path,
                                                compile_model=compile_model, amp_precision=amp_precision)
        self.clip = clip.to(self.device)
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()
        self.latent_std = latent_std.to(self.device)
        self.latent_mean = latent_mean.to(self.device)
        self.autoencoder = autoencoder.to(self.device)
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()
        self.tokenizer = tokenizer

    def preprocessing_data(self, data):
        latent = data[0].to(self.device)
        token = data[1].to(self.device)
        latent = (latent - self.latent_mean) / self.latent_std
        with torch.no_grad():
            token_global, token_pool = self.clip.text_encoder.encode_text(token)
        return latent, token_global, token_pool

    def compute_loss(self, data):
        loss = self.model.loss(data[0], data[1], data[2])
        self.validate_loss(loss=loss, optimizer=self.optimizer)
        return loss

    def validate(self, epoch, **kwargs) -> None:
        assert "latent_shape" in list(kwargs.keys()), "必须显式传入latent_shape参数"
        assert "valid_data" in list(kwargs.keys()), "必须显式输入valid_data参数"
        image_shape = kwargs["latent_shape"]
        valid_data = kwargs["valid_data"]
        file_path = os.path.join(self.valid_path, f"{epoch}.jpg")
        valid_batch_size = kwargs.get("valid_batch_size", 4)
        valid_data = [valid_data] * valid_batch_size
        token = self.tokenizer(valid_data)
        row = int(math.sqrt(valid_batch_size))
        if self.model.__class__.__name__ == "Diffusion":
            mode = kwargs.get("mode", "dpm")
        else:
            mode = kwargs.get("mode", "euler")
        self.model.eval()
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.model)
        with torch.no_grad():
            token_global, token_pool = self.clip.text_encoder.encode_text(token)
            latent = self.model.sample(batch_size=valid_batch_size, mode=mode, image_shape=image_shape,
                                       condition1=token_global, condition2=token_pool).clamp(min=-1, max=1)
            latent = latent * self.latent_std + self.latent_mean
            sample = self.autoencoder.latent2image(latent).clamp(min=-1, max=1)
            save_image((sample.to(torch.float32) + 1) / 2, fp=file_path, nrow=row,
                       normalize=False, padding=1)
        if self.using_ema:
            self.ema.restore(model=self.model, backup=back_params)
        self.model.train()
