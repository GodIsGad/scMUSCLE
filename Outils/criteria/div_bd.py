import torch
import Outils.batchminer as batchminer
from Outils.criteria import binomial_deviance as bd

ALLOWED_MINING_OPS = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM = False


class DivBD(torch.nn.Module):
    def __init__(self, opt):
        super(DivBD, self).__init__()
        self.opt = opt

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def forward(self, batch, labels, **kwargs):

        if len(batch.shape) == 3 and batch.shape[1] > 1:
            return bd.dirdiv_bd(batch, self.opt)
        else:
            return torch.tensor(0.0).cuda()

