from ..losses import *


class ESRLoss(nn.Module):
    def __init__(self, using_perception: bool = False, perception_weight=1., perception_net="alex",
                 device=None):
        super(ESRLoss, self).__init__()
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
