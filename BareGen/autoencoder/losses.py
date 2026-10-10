from .structs import *
from ..losses import l1, l2


class AutoEncoderKLLoss(nn.Module):
    def __init__(self, using_perception: bool = False, perception_weight=1., perception_net="alex",
                 kl_weight=1e-3, device=None):
        super(AutoEncoderKLLoss, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rec_loss = l1
        self.kl_weight = kl_weight
        self.using_perception = using_perception
        self.perception_weight = perception_weight
        if self.using_perception:
            self.perception_loss = LPIPS(net=perception_net).to(device).eval()
            for param in self.perception_loss.parameters():
                param.requires_grad = False

    @autocast(device_type="cuda", enabled=False)
    def forward(self, inputs, targets, latents: DiagonalGaussianDistribution):
        batch_size = inputs.size(0)
        rec_loss = self.rec_loss(inputs, targets)
        kl = latents.kl()
        if self.using_perception:
            p_loss = self.perception_loss(inputs, targets)
            rec_loss = rec_loss + self.perception_weight * p_loss
        rec_loss = rec_loss.sum() / batch_size
        kl_loss = kl.sum() / batch_size
        return rec_loss + self.kl_weight * kl_loss


class VQAutoEncoderLoss(nn.Module):
    def __init__(self, using_perception: bool = False, perception_weight=1., perception_net="alex",
                 book_weight=1., device=None):
        super(VQAutoEncoderLoss, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rec_loss = l1
        self.book_weight = book_weight
        self.using_perception = using_perception
        self.perception_weight = perception_weight
        if self.using_perception:
            self.perception_loss = LPIPS(net=perception_net).to(device).eval()
            for param in self.perception_loss.parameters():
                param.requires_grad = False

    @autocast(device_type="cuda", enabled=False)
    def forward(self, inputs, targets, book_loss: torch.Tensor):
        rec_loss = self.rec_loss(inputs, targets)
        if self.using_perception:
            p_loss = self.perception_loss(inputs, targets)
            rec_loss = rec_loss + self.perception_weight * p_loss
        rec_loss = rec_loss.mean()
        return rec_loss + self.book_weight * book_loss


class PlainAutoEncoderLoss(nn.Module):
    def __init__(self, using_perception: bool = False, perception_weight=1., perception_net="alex",
                 device=None):
        super(PlainAutoEncoderLoss, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rec_loss = l1
        self.using_perception = using_perception
        self.perception_weight = perception_weight
        if self.using_perception:
            self.perception_loss = LPIPS(net=perception_net).to(device).eval()
            for param in self.perception_loss.parameters():
                param.requires_grad = False

    @autocast(device_type="cuda", enabled=False)
    def forward(self, inputs, targets):
        rec_loss = self.rec_loss(inputs, targets)
        if self.using_perception:
            p_loss = self.perception_loss(inputs, targets)
            rec_loss = rec_loss + self.perception_weight * p_loss
        rec_loss = rec_loss.mean()
        return rec_loss
