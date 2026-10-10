from .import_packages import *


def l1(inputs, targets):
    return torch.abs(targets - inputs)


def l2(inputs, targets):
    return torch.square(targets - inputs)


@autocast(device_type="cuda", enabled=False)
def hinge(inputs, targets):
    loss_real = torch.mean(F.relu(1. - targets))
    loss_fake = torch.mean(F.relu(1. + inputs))
    d_loss = loss_real + loss_fake
    return d_loss

@autocast(device_type="cuda", enabled=False)
def discriminator_gradient_penalty_loss(discriminator, inputs,
                                        targets, create_graph=True,
                                        retain_graph=True, lambda_gp=10):
    alpha = torch.rand(size=(targets.shape[0], 1, 1, 1)).to(targets.device)
    interpolates = (alpha * targets + ((1 - alpha) * inputs)).requires_grad_(True).to(targets.device)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(size=(targets.shape[0], 1), requires_grad=False).to(targets.device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    true_score = discriminator(targets)
    fake_score = discriminator(inputs.detach())
    return -true_score.mean() + fake_score.mean() + lambda_gp * gradient_penalty
