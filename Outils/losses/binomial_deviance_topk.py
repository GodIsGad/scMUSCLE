import torch
import torch.nn.functional as F
import sys
sys.path.append('/mnt/dive/shared/xyxu/projects/ReDML-icml20')


class BinomialDevianceTopk(torch.nn.Module):
    def __init__(self, args):
        super(BinomialDevianceTopk, self).__init__()
        self.lam_glb = args.lam_glb
        self.add_global_group = args.add_global_group
        self.device = args.device
        self.aug = args.aug
        self.K = 200

    def forward(self, xs, x_augs, *args):
        xs = F.normalize(xs, p=2, dim=-1)
        x_augs = F.normalize(x_augs, p=2, dim=-1)
        loss_local = 0.0

        _, k, _ = xs.size()
        ng = k-1 if self.add_global_group else k

        for i in range(ng):
            loss_local += self.bd_loss(xs[:, i, :], x_augs[:, i, :])

        loss = loss_local / torch.tensor(ng, dtype=torch.float32)

        # global
        if self.add_global_group:
            loss_global = self.bd_loss(xs[:, -1, :], x_augs[:, -1, :])
            loss += self.lam_glb * loss_global
        return loss

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

        sim = torch.matmul(x, x_aug.t())

        N = x.shape[0]

        pos_mask = torch.eye(N).to(self.device)
        neg_mask = torch.ones((N, N)).to(self.device) - pos_mask
        if self.aug[0] == 'none' and self.aug[1] == 'none':
            neg_mask = torch.triu(neg_mask)

        cons = -1.0 * pos_mask + C * neg_mask

        act = alpha * (sim - beta) * cons

        loss = torch.log(torch.exp(act) + torch.tensor(1.0, dtype=data_type).to(self.device))
        pos_loss = torch.masked_select(loss, pos_mask.bool())
        neg_loss = torch.masked_select(loss, neg_mask.bool())

        loss_all = torch.cat([pos_loss, neg_loss])
        top_loss, _ = torch.topk(loss_all, self.K, largest=True)
        loss = torch.mean(top_loss)

        return torch.sum(loss)