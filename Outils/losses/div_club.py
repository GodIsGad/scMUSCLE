import torch
import torch.nn.functional as F
import sys

sys.path.append('/data/xxy/my-DML-projects/ReML-icml20')
sys.path.append('/home/wjx/GroupCL/Outils/CLUB')
from Outils.CLUB.mi_estimators import CLUB, CLUBSample
from pytorch_revgrad import RevGrad


class DivCLUB(torch.nn.Module):
    def __init__(self, args):
        super(DivCLUB, self).__init__()
        self.lam_glb = args.lam_glb
        self.add_global_group = args.add_global_group
        self.device = args.device

        self.k = args.num_group
        self.embedding_dim = args.embedding_dim
        self.dim_per_group = self.embedding_dim // self.k
        # self.embedding_dim = self.k * self.dim_per_group
        self.club_type = args.club_type  # 'CLUB' or 'CLUBSample'

        self.grad_reverse = RevGrad()

        self.club_hidden = args.club_hidden
        self.clubs = torch.nn.ModuleList()
        self.club_fea_norm = args.club_fea_norm

        self.club_type = CLUB if args.club_type == 'CLUB' else CLUBSample

        for i in range(self.k):
            for j in range(i+1, self.k):
                self.clubs.append(self.club_type(x_dim=self.dim_per_group, y_dim=self.dim_per_group,
                                  hidden_size=self.club_hidden, args=args).to(self.device))

    def forward(self, xs, *args):
        if self.club_fea_norm == 'true':
            xs = F.normalize(xs, p=2, dim=-1)
        loss_div = 0.0
        _, k, _ = xs.size()
        ng = k-1 if self.add_global_group else k
        assert ng > 1, 'No. of groups should larger than one'
        count = 0
        for i in range(ng):
            for j in range(i+1, ng):
                loss_div += self.clubs[count](xs[:, i, :], xs[:, j, :])
                count += 1
        return loss_div

    def club_params_loss(self, xs):
        if self.club_fea_norm == 'true':
            xs = F.normalize(xs, p=2, dim=-1)
        loss_club = 0.0
        _, k, _ = xs.size()
        ng = k - 1 if self.add_global_group else k
        assert ng > 1, 'No. of groups should larger than one'
        count = 0
        for i in range(ng):   # 遍历所有可能的组组合，并计算它们之间的某种损失
            for j in range(i + 1, ng):
                loss_club += self.clubs[count].learning_loss(xs[:, i, :], xs[:, j, :])
                count += 1
        return loss_club

