import torch, torch.nn.functional as F

ALLOWED_MINING_OPS = ['npair']
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = False

def dirdiv_bd(batch, opt=None):
    bd_loss_param = [2.0, 0.5, 25]
    if opt is not None:
        bd_loss_param[2] = opt.cn
    loss_div = 0.
    n, k, dim = batch.size()

    pos_mask = torch.zeros([n, n], dtype=torch.float32).cuda()
    neg_mask = torch.eye(n, dtype=torch.float32).cuda()
    for i in range(k):
        for j in range(i + 1, k):
            loss_div += bd_loss(batch[:, i, :], batch[:, j, :], pos_mask, neg_mask, bd_loss_param)
    if k > 1:
        loss_div = loss_div / (k-1)

    return loss_div

def bd_loss(fea1, fea2, pos_mask, neg_mask, bd_loss_param=(2.0, 0.5, 25)):
    beta1, beta2, C = bd_loss_param
    data_type = torch.float32
    n_fea1, n_fea2 = F.normalize(fea1, p=2, dim=-1), F.normalize(fea2, p=2, dim=-1)
    sim = torch.matmul(n_fea1, n_fea2.transpose(1, 0))

    cons = -1.0 * pos_mask + C * neg_mask
    act = beta1 * (sim - beta2) * cons

    a = torch.tensor(1e-5, dtype=data_type).to('cuda')
    norm_mask = pos_mask / torch.max(a, torch.sum(pos_mask)) + \
                neg_mask / torch.max(a, torch.sum(neg_mask))
    loss_vec = torch.log(torch.exp(act) + 1.0) * norm_mask
    return torch.sum(loss_vec)


def pair_mask(self_compare=False, *lab):
    data_type = torch.float32
    cla_lab1, cla_lab2 = lab[:2]
    pos_pair_lab = torch.eq(cla_lab1.view(-1, 1), cla_lab2.view(1, -1))
    if len(lab) == 4:
        pro_lab1, pro_lab2 = lab[2:]
        x = torch.eq(pro_lab1.view(-1, 1), pro_lab2.view(1, -1))
        pos_pair_lab = pos_pair_lab * x  # pos condition: cla_lab1==cla_lab2 && pro_lab1==pro_lab2
    if self_compare is True:
        n = cla_lab1.size()[0]
        w = torch.ones([n, n], dtype=data_type) - torch.eye(n, dtype=data_type)
        pos_pair_lab *= w
    pos_mask = pos_pair_lab.float()

    neg_cla_pair_lab = torch.eq(cla_lab1.view(-1, 1), cla_lab2.view(1, -1))
    neg_mask = torch.tensor(1, dtype=data_type) - neg_cla_pair_lab
    return pos_mask.cuda(), neg_mask.cuda()


class Criterion(torch.nn.Module):
    def __init__(self, opt, self_compare=False, bd_loss_param=(2.0, 0.5, 25)):
        super(Criterion, self).__init__()
        self.self_compare = self_compare
        self.bd_loss_param = bd_loss_param
        self.opt = opt

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def forward(self, batch, labels, **kwargs):
        cuda = torch.device('cuda')
        pos_mask, neg_mask = pair_mask(False, labels, labels)
        loss_emb, loss_div = torch.tensor(0.0, device=cuda), torch.tensor(0.0, device=cuda)
        if len(batch.size()) > 2:
            n, k, dim = batch.size()
            for i in range(k):
                loss_emb += bd_loss(batch[:, i, :], batch[:, i, :], pos_mask, neg_mask)
            loss_emb /= k
            loss_div += eval('diversity.{}(batch, self.opt)'.format(self.opt.diversity))
        else:
            loss_emb = bd_loss(batch, batch, pos_mask, neg_mask)

        loss = self.opt.lam_emb * loss_emb + self.opt.lam_div * loss_div
        return loss, loss_emb, loss_div

