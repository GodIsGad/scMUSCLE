import torch
import torch.nn.functional as F


class DivKL(torch.nn.Module):
    def __init__(self, args):
        super(DivKL, self).__init__()
        self.lam_glb = args.lam_glb
        self.add_global_group = args.add_global_group
        self.device = args.device

    def forward(self, xs, *args):
        loss_div = 0.0
        _, k, _ = xs.size()
        ng = k - 1 if self.add_global_group else k
        if ng == 1:
            return torch.tensor(0.0).to('cuda')
        else:
            assert ng > 1, 'No. of groups should larger than one'
            for i in range(ng):
                for j in range(i + 1, ng):
                    loss_div += self.kl_loss(xs[:, i, :], xs[:, j, :])
            loss = loss_div / torch.tensor(ng - 1, dtype=torch.float32)
            return loss

    def kl_loss(self, x_g1, x_g2):
        x_g1_norm = F.softmax(x_g1, dim=-1)
        x_g2_norm = F.softmax(x_g2, dim=-1)
        kl_loss = torch.mean(torch.sum(x_g1_norm * (x_g1_norm / x_g2_norm).log(), dim=-1), dim=0)
        return -kl_loss