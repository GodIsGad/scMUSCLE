import torch
import torch.nn.functional as F
import sys
sys.path.append('/mnt/dive/shared/xyxu/projects/ReDML-icml20')
from Outils.criteria import binomial_deviance as bd
#


class SoftBinomialDevianceLoss(torch.nn.Module):
    def __init__(self, args):
        super(SoftBinomialDevianceLoss, self).__init__()
        self.k = args.num_group
        self.lam_div = args.lam_div
        self.lam_glb = args.lam_glb
        self.add_global_group = args.add_global_group
        self.margin = args.loss_soft_margin
        self.beta = torch.nn.Parameter(torch.ones((self.k, args.n_classes)) * args.loss_soft_beta)

    def forward(self, xs, x_augs, *vars):
        if self.add_global_group is True:
            xs, xs_global = xs
            x_augs, x_augs_global = x_augs

        xs = F.normalize(xs, p=2, dim=-1)
        x_augs = F.normalize(x_augs, p=2, dim=-1)
        loss_emb, loss_global, loss_div = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        _, k, _ = xs.size()
        # y = torch.cat([torch.arange(xs.shape[0]), torch.arange(xs.shape[0])], dim=0)
        y = vars[0]
        y2 = torch.cat([y, y])
        for i in range(k):
            beta = self.beta[i][y2]
            x, x_aug = xs[:, i, :], x_augs[:, i, :]
            loss_emb += self.bd_loss(torch.cat([x, x_aug], dim=0), y2.cuda(), beta)

        loss_emb /= torch.tensor(k, dtype=torch.float32)
        loss_div += bd.dirdiv_bd(xs)
        loss_div += bd.dirdiv_bd(x_augs)
        loss_div /= 2

        loss = loss_emb + self.lam_div * loss_div

        # global
        if self.add_global_group is True:
            loss_global += self.bd_loss(torch.cat([xs_global, x_augs_global], dim=0), y.cuda())
            loss += self.lam_glb * loss_global
        return loss, loss_emb, loss_div, loss_global

    def bd_loss(self, x, y, soft_margin):
        """
        Args:
            x: num_graph * fea_dim
            y: num_graph * 1
            param: bd loss params
        Returns: loss
        """
        soft_margin_matrix = self.set_soft_margin(soft_margin)
        margin = self.margin
        alpha, C = 2.0, 1.0
        margin_matrix = margin

        data_type = torch.float32
        norm_x = F.normalize(x, p=2, dim=-1)
        sim = torch.matmul(norm_x, norm_x.transpose(1, 0))

        pos_mask = (torch.eq(y.view(-1, 1), y.view(1, -1))).float() - torch.eye(len(y)).cuda()
        neg_mask = torch.ones([len(y), len(y)]).cuda() - \
                   (torch.eq(y.view(-1, 1), y.view(1, -1))).float()

        norm_mask = pos_mask / torch.max(torch.tensor(1e-5, dtype=data_type).cuda(), torch.sum(pos_mask)) + \
                    neg_mask / torch.max(torch.tensor(1e-5, dtype=data_type).cuda(), torch.sum(neg_mask))

        cons = -1.0 * pos_mask + C * neg_mask
        act = alpha * (sim - margin_matrix) * cons
        loss = torch.log(torch.exp(act) + torch.tensor(1.0, dtype=data_type).cuda()) * norm_mask
        return torch.sum(loss)

    def set_soft_margin(self, soft_margin):
        n = len(soft_margin)
        expand1 = (soft_margin.unsqueeze(1)).repeat(1, n)
        # expand2 = (soft_margin.unsqueeze(0)).repeat(n, 1)
        return expand1
