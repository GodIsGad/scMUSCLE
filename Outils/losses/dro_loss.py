import torch
import torch.nn.functional as F
import sys


# sys.path.append('/data/xxy/my-DML-projects/ReML-icml20')
# sys.path.append('/home/wjx/GroupCL/Outils/DRO')
sys.path.append('/home/zwz/jyz/scMCs_Desktop/Outils/DRO')
from Outils.DRO.DRO_TOPK import DRO_TOPK


class DroLoss(torch.nn.Module):
    def __init__(self, args):
        super(DroLoss, self).__init__()
        self.lam_glb = args.lam_glb
        self.add_global_group = args.add_global_group
        self.device = args.device

        self.k = args.num_group
        self.embedding_dim = args.embedding_dim
        self.dim_per_group = self.embedding_dim // self.k
        self.embedding_dim = self.k * self.dim_per_group

        self.club_hidden = args.club_hidden
        self.emb_loss = DRO_TOPK().to(self.device)

    def forward(self, xs, x_augs, *args):
        xs = F.normalize(xs, p=2, dim=-1)
        x_augs = F.normalize(x_augs, p=2, dim=-1)
        loss = 0.0

        n, k, _ = xs.size()
        ng = k-1 if self.add_global_group else k

        label = torch.arange(n).to(self.device)
        for i in range(ng):
            x = xs[:, i, :]
            x_aug = x_augs[:, i, :]
            loss += self.emb_loss(torch.cat([x, x_aug], dim=0), torch.cat([label, label]))[0]

        loss = loss / torch.tensor(ng, dtype=torch.float32)
        return loss

