import torch.nn as nn
import torch.nn.functional as F


class NegativeLogLikelihood(nn.Module):
    def __init__(self, args):
        super(NegativeLogLikelihood, self).__init__()
        self.k = args.num_group

    def forward(self, x, y):
        loss=0
        for i in range(self.k):
            loss += F.nll_loss(x[:, i, :], y)

        loss /= self.k
        return loss
