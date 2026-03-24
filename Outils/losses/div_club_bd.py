import torch
import torch.nn.functional as F
import sys
sys.path.append('/mnt/dive/shared/xyxu/projects/ReDML-icml20')


class DivClubBD(torch.nn.Module):
    def __init__(self, args):
        super(DivClubBD, self).__init__()
        self.lam_glb = args.lam_glb
        self.add_global_group = args.add_global_group
        self.device = args.device
        self.pos_pairs = args.pos_pairs

    def forward(self, xs, *args):
        xs = F.normalize(xs, p=2, dim=-1)
        loss_div = 0.0
        _, k, _ = xs.size()
        ng = k-1 if self.add_global_group else k
        assert ng > 1, 'No. of groups should larger than one'
        for i in range(ng):
            for j in range(i+1, ng):
                loss_div += self.bd_loss(xs[:, i, :], xs[:, j, :])
        loss = loss_div / torch.tensor(ng-1, dtype=torch.float32)
        return loss

    def bd_loss(self, x_g1, x_g2, param=(2.0, 0.5, 1)):
        """
        Args:
            x: num_graph * fea_dim
            y: num_graph * 1
            param: bd loss params
        Returns: loss
        """
        alpha, beta, C = param
        data_type = torch.float32

        sim = torch.matmul(x_g1, x_g2.t())

        N = x_g1.shape[0]

        neg_mask = torch.eye(N).to(self.device)
        if self.pos_pairs == 'true':
            pos_mask = torch.ones((N, N)).to(self.device) - neg_mask
        else:
            pos_mask = torch.zeros((N, N)).to(self.device)

        cons = -1.0 * pos_mask + C * neg_mask

        # act = alpha * (sim - beta) * cons
        a = torch.tensor(1e-5, dtype=data_type).to(self.device)
        norm_mask = pos_mask / torch.max(a, torch.sum(pos_mask)) + \
                    neg_mask / torch.max(a, torch.sum(neg_mask))
        # loss = torch.log(torch.exp(act) + torch.tensor(1.0, dtype=data_type).to(self.device)) * norm_mask
        loss = sim * cons * norm_mask

        return torch.sum(loss)