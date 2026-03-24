import torch
import torch.nn.functional as F


class DivBD(torch.nn.Module):
    def __init__(self, args):
        super(DivBD, self).__init__()
        self.lam_glb = args.lam_glb
        self.add_global_group = args.add_global_group
        self.device = 1

    def forward(self, xs, *args):
        xs = F.normalize(xs, p=2, dim=-1)
        loss_div = 0.0
        _, k, _ = xs.size()
        ng = k-1 if self.add_global_group else k
        if ng == 1:
            return torch.tensor(0.0).to(self.device)
        else:
            for i in range(ng):
                for j in range(i+1, ng):
                    loss_div += self.bd_loss(xs[:, i, :], xs[:, j, :])
            loss = loss_div / torch.tensor(ng-1, dtype=torch.float32)
            return loss

    def bd_loss(self, x_g1, x_g2, param=(2.0, 0.5, 25)):
        alpha, beta, C = param
        data_type = torch.float32

        sim = torch.matmul(x_g1, x_g2.t())

        N = x_g1.shape[0]

        neg_mask = torch.eye(N).to(self.device)
        pos_mask = torch.zeros((N, N)).to(self.device)

        cons = -1.0 * pos_mask + C * neg_mask

        act = alpha * (sim - beta) * cons
        a = torch.tensor(1e-5, dtype=data_type).to(self.device)
        norm_mask = pos_mask / torch.max(a, torch.sum(pos_mask)) + \
                    neg_mask / torch.max(a, torch.sum(neg_mask))
        loss = torch.log(torch.exp(act) + torch.tensor(1.0, dtype=data_type).to(self.device)) * norm_mask
        return torch.sum(loss)