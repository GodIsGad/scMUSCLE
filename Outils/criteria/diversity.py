import torch, torch.nn.functional as F
# from Outils.criteria.binomial_deviance import bd_loss
from .binomial_deviance import bd_loss

def dirdiv_bd(batch, opt=None):
    bd_loss_param = [2.0, 0.5, 25]
    if opt is not None:
        bd_loss_param[2] = opt.cn
    loss_div = 0.
    n, k, dim = batch.size()

    pos_mask = torch.zeros([n, n], dtype=torch.float32).cuda()
    neg_mask = torch.eye(n, dtype=torch.float32).cuda()
    for i in range(k):
        for j in range(i + 1, k):
            loss_div += bd_loss(batch[:, i, :], batch[:, j, :], pos_mask, neg_mask, bd_loss_param)
    if k > 1:
        loss_div = loss_div / (k-1)

    return loss_div