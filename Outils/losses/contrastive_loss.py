import torch.nn as nn
import torch.nn.functional as F
import torch


class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()

    def forward(self, x1, x2):
        T = 0.5
        batch_size, _ = x1.size()

        # batch_size *= 2
        # x1, x2 = torch.cat((x1, x2), dim=0), torch.cat((x2, x1), dim=0)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        '''
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        self_sim = sim_matrix[range(batch_size), list(range(int(batch_size/2), batch_size))+list(range(int(batch_size/2)))]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim - self_sim)
        loss = - torch.log(loss).mean()
        '''

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss
