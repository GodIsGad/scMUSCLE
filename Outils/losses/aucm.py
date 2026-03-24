import torch
import torch.nn.functional as F
import sys
sys.path.append('/mnt/dive/shared/xyxu/projects/ReDML-icml20')
from libauc.losses import AUCMLoss


class AUCM(torch.nn.Module):
    def __init__(self, args):
        super(AUCM, self).__init__()
        self.lam_glb = args.lam_glb
        self.device = args.device
        imratio = 1 / (args.batch_size - 1)
        self.loss = AUCMLoss(imratio=imratio)

    def forward(self, xs, x_augs, *args):
        xs = F.normalize(xs, p=2, dim=-1)
        x_augs = F.normalize(x_augs, p=2, dim=-1)

        loss_local = 0.0

        n, k, _ = xs.size()
        ng = k - 1 if self.add_global_group else k

        for i in range(ng):
            x, x_aug = xs[:, i, :], x_augs[:, i, :]
            sim = torch.einsum('ik, jk->ij', x, x_augs)
            label = torch.eye(n).cuda()
            loss_local += self.loss(sim, label)

        loss = loss_local / torch.tensor(ng, dtype=torch.float32)

        # global
        if self.add_global_group:
            loss_global = self.bd_loss(xs[:, -1, :], x_augs[:, -1, :])
            loss += self.lam_glb * loss_global
        return loss