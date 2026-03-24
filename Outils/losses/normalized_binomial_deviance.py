import torch
import torch.nn.functional as F
import sys
sys.path.append('/mnt/dive/shared/xyxu/projects/ReDML-icml20')
from Outils.criteria import binomial_deviance as bd


class NormalizedBinomialDevianceLoss(torch.nn.Module):
    def __init__(self, args):
        super(NormalizedBinomialDevianceLoss, self).__init__()
        self.lam_div = args.lam_div
        self.lam_glb = args.lam_glb
        self.add_global_group = args.add_global_group
        self.device = args.device

    def forward(self, xs, x_augs, *args):
        if self.add_global_group is True:
            xs, xs_global = xs
            x_augs, x_augs_global = x_augs

        xs = F.normalize(xs, p=2, dim=-1)
        x_augs = F.normalize(x_augs, p=2, dim=-1)
        loss_emb, loss_div = 0.0, 0.0
        loss_global = torch.tensor(0.).to(self.device)

        _, k, _ = xs.size()
        for i in range(k):
            x, x_aug = xs[:, i, :], x_augs[:, i, :]
            loss_emb += self.bd_loss(x, x_aug)

        loss_emb /= torch.tensor(k, dtype=torch.float32)
        loss_div += bd.dirdiv_bd(xs, device=self.device)
        loss_div += bd.dirdiv_bd(x_augs)
        loss_div /= 2

        loss = loss_emb + self.lam_div * loss_div

        # global
        if self.add_global_group is True:
            loss_global += self.bd_loss(xs_global, x_augs_global)
            loss += self.lam_glb * loss_global
        return loss, loss_emb, loss_div, loss_global

    def bd_loss(self, x, x_aug, param=(2.0, 0.5, 25)):
        """
        Args:
            x: num_graph * fea_dim
            y: num_graph * 1
            param: bd loss params
        Returns: loss
        """
        alpha, beta, C = param
        data_type = torch.float32
        norm_x = F.normalize(x, p=2, dim=-1)
        norm_x_aug = F.normalize(x_aug, p=2, dim=-1)
        sim = torch.matmul(norm_x, norm_x_aug.t())

        N = x.shape[0]

        pos_mask = torch.eye(N).to(self.device)
        neg_mask = torch.ones([N, N]).to(self.device) - pos_mask

        cons = -1.0 * pos_mask + C * neg_mask

        act = alpha * (sim - beta) * cons
        a = torch.tensor(1e-5, dtype=data_type).to(self.device)
        norm_mask = pos_mask / torch.max(a, torch.sum(pos_mask)) + \
                    neg_mask / torch.max(a, torch.sum(neg_mask))
        loss = torch.log(torch.exp(act) + torch.tensor(1.0, dtype=data_type).to(self.device)) * norm_mask
        return torch.sum(loss)