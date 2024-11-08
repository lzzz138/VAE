import torch
import torch.nn.functional as F

def loss_function(x_hat, x, u, log_var):
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(u, 2) - 1. - log_var)
    loss = BCE + KLD
    return loss, BCE, KLD