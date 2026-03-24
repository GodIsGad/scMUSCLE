import torch
import torch.nn.functional as F
# import sys
# sys.path.append('/mnt/dive/shared/xyxu/projects/ReDML-icml20')


class BinomialDevianceLoss(torch.nn.Module):
    def __init__(self, args):
        super(BinomialDevianceLoss, self).__init__()
        self.lam_glb = args.lam_glb
        self.add_global_group = args.add_global_group
        self.device = 1

    def forward(self, xs, x_augs, *args):
        xs = F.normalize(xs, p=2, dim=-1)
        x_augs = F.normalize(x_augs, p=2, dim=-1)
        loss_local = 0.0

        _, k, _ = xs.size()
        ng = k-1 if self.add_global_group else k

        for i in range(ng):
            # loss_local += self.bd_loss(xs[:, i, :], x_augs[:, i, :])
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

        cons = -1.0 * pos_mask + C * neg_mask

        act = alpha * (sim - beta) * cons
        a = torch.tensor(1e-5, dtype=data_type).to(self.device)
        norm_mask = pos_mask / torch.max(a, torch.sum(pos_mask)) + \
                    neg_mask / torch.max(a, torch.sum(neg_mask))
        loss = torch.log(torch.exp(act) + torch.tensor(1.0, dtype=data_type).to(self.device)) * norm_mask
        return torch.sum(loss)


class WeightedBDLoss(torch.nn.Module):
    def __init__(self, args):
        super(WeightedBDLoss, self).__init__()
        self.lam_glb = args.lam_glb
        self.add_global_group = args.add_global_group
        self.device = args.device

    def forward(self, xs, x_augs, *args):
        xs = F.normalize(xs, p=2, dim=-1)
        x_augs = F.normalize(x_augs, p=2, dim=-1)
        loss = 0.0

        _, k, _ = xs.size()

        weights = self.compute_weights_for_PP(xs, x_augs) # N * K
        for i in range(k):
            loss += self.neg_loss(xs[:, i, :], x_augs[:, i, :])
            loss += self.pos_loss(xs[:, i, :], x_augs[:, i, :], weights[:, i])
        loss = loss / torch.tensor(k, dtype=torch.float32)
        return loss

    def neg_loss(self, x, x_aug, param=(2.0, 0.5, 25)):
        alpha, beta, C = param
        n, _ = x.shape
        sim = torch.matmul(x, x_aug.t())
        act = (sim - beta) * alpha * C

        neg_mask = torch.ones([n, n]) - torch.eye(n)
        neg_mask = neg_mask / torch.sum(neg_mask)

        neg_loss = torch.log(torch.exp(act) + 1.) * neg_mask.to(self.device)
        return torch.sum(neg_loss)

    def pos_loss(self, x, x_aug, weights, param=(2.0, 0.5, 1)):
        alpha, beta, C = param
        n, _ = x.shape
        sim = torch.sum(x * x_aug, dim=-1)
        act = -(sim - beta) * alpha * C

        pos_loss = torch.log(torch.exp(act) + 1.) / n * weights.cuda()
        return torch.sum(pos_loss)

    def compute_weights_for_PP(self, xs, x_augs):
        n, k, _ = xs.shape
        xs = xs.cpu().detach()
        x_augs = x_augs.cpu().detach()
        sim = []
        for i in range(k):
            pos_pairs_sim = torch.sum(xs[:, i, :]*x_augs[:, i, :], dim=-1, keepdim=True)
            sim.append(pos_pairs_sim)
        sim = torch.cat(sim, dim=-1)
        # sim = F.softmax(sim, dim=-1)
        # sim = torch.ones_like(sim)
        sim = torch.sigmoid(sim)
        # sim = F.normalize(sim, p=2, dim=-1)
        return sim


class InfoGraphBDLoss(torch.nn.Module):
    def __init__(self, args):
        super(InfoGraphBDLoss, self).__init__()
        self.lam_glb = args.lam_glb
        self.add_global_group = args.add_global_group
        self.device = args.device
        self.top_k = args.top_k

    def forward(self, z_g, z_n, batch):
        #normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12,
                      # out: Optional[Tensor] = None) -> Tensor:
        z_g = F.normalize(z_g, p=2, dim=-1)
        z_n = F.normalize(z_n, p=2, dim=-1)

        _, k, _ = z_g.size()
        ng = k - 1 if self.add_global_group else k
        loss_local = 0.0

        for i in range(ng):
            loss_local += self.bd_loss(z_g[:, i, :], z_n, batch, top_k=self.top_k)

        loss = loss_local / torch.tensor(ng, dtype=torch.float32)

        return loss

    def bd_loss(self, z, z_n, batch, top_k, param=(2.0, 0.5, 25)):
        """
        Args:
            x: num_graph * fea_dim
            y: num_graph * 1
            param: bd loss params
        Returns: loss
        """
        loss = 0.0

        loss += self.neg_loss_graph(z)
        # loss += self.neg_loss_nodes(z, z_n, batch)
        loss += self.pos_loss(z, z_n, batch, top_k)
        return loss

    def neg_loss_graph(self, z, param=(2.0, 0.5, 25)):

        alpha, beta, C = param
        data_type = torch.float32
        sim = torch.matmul(z, z.t())
        N = z.shape[0]
        y = torch.arange(N).cuda()

        a = torch.tensor(1e-5, dtype=data_type).to(self.device)
        # neg_loss
        neg_mask = 1. - torch.eq(y.unsqueeze(1), y.unsqueeze(0)).float()
        neg_act = alpha * (sim - beta) * C
        norm_neg_mask = neg_mask / torch.max(a, torch.sum(neg_mask))
        neg_loss = torch.log(torch.exp(neg_act) + torch.tensor(1.0, dtype=data_type).to(self.device)) * norm_neg_mask
        return torch.sum(neg_loss)

    def neg_loss_nodes(self, z, z_n, batch, param=(2.0, 0.5, 25)):

        alpha, beta, C = param
        data_type = torch.float32
        sim = torch.matmul(z, z_n.t())
        N = z.shape[0]
        y = torch.arange(N).cuda()

        a = torch.tensor(1e-5, dtype=data_type).to(self.device)
        # neg_loss
        neg_mask = 1. - torch.eq(y.unsqueeze(1), batch.unsqueeze(0)).float()
        neg_act = alpha * (sim - beta) * C
        norm_neg_mask = neg_mask / torch.max(a, torch.sum(neg_mask))
        neg_loss = torch.log(torch.exp(neg_act) + torch.tensor(1.0, dtype=data_type).to(self.device)) * norm_neg_mask
        return torch.sum(neg_loss)

    def pos_loss(self, z, z_n, batch, top_k, param=(2.0, 0.5, 25)):
        alpha, beta, C = param
        data_type = torch.float32
        a = torch.tensor(1e-5, dtype=data_type).to(self.device)

        N = z.shape[0]
        y = torch.arange(N).cuda()
        sim = torch.matmul(z, z_n.t())

        pos_mask, sorted_pos_sim = self.get_pos_mask(y, y_n=batch, sim=sim, top_k=top_k)
        pos_act = -alpha * (sorted_pos_sim - beta)
        norm_pos_mask = pos_mask / torch.max(a, torch.sum(pos_mask))
        pos_loss = torch.log(torch.exp(pos_act) + torch.tensor(1.0, dtype=data_type).to(self.device)) * norm_pos_mask
        return torch.sum(pos_loss)

    def get_pos_mask(self, y, y_n, sim, top_k=5):
        pos_mask = torch.eq(y.unsqueeze(1), y_n.unsqueeze(0)).float()
        pos_sim = sim * pos_mask
        # pos_sim, _ = torch.sort(pos_sim, descending=True)
        #
        # pos_mask = torch.zeros_like(sim)
        # pos_mask[:, :top_k] = 1.
        return pos_mask, pos_sim


class BDLoss(torch.nn.Module):
    def __init__(self, args):
        super(BDLoss, self).__init__()
        self.device = args.device

    def forward(self, xs, y):
        xs = F.normalize(xs, p=2, dim=-1)
        loss_local = 0.0

        _, ng, _ = xs.size()

        for i in range(ng):
            loss_local += self.bd_loss(xs[:, i, :], y)

        loss = loss_local / torch.tensor(ng, dtype=torch.float32)

        return loss

    def bd_loss(self, x, y, param=(2.0, 0.5, 25)):
        """
        Args:
            x: num_graph * fea_dim
            y: num_graph * 1
            param: bd loss params
        Returns: loss
        """
        alpha, beta, C = param
        data_type = torch.float32

        sim = torch.matmul(x, x.t())

        N = x.shape[0]

        pos_mask = torch.eq(y.unsqueeze(1), y.unsqueeze(0)).float()
        neg_mask = torch.ones((N, N)).to(self.device) - pos_mask
        pos_mask -= torch.eye(N)

        pos_mask, neg_mask = pos_mask.cuda(), neg_mask.cuda()

        cons = -1.0 * pos_mask + C * neg_mask

        act = alpha * (sim - beta) * cons
        a = torch.tensor(1e-5, dtype=data_type).to(self.device)
        norm_mask = pos_mask / torch.max(a, torch.sum(pos_mask)) + \
                    neg_mask / torch.max(a, torch.sum(neg_mask))
        loss = torch.log(torch.exp(act) + torch.tensor(1.0, dtype=data_type).to(self.device)) * norm_mask
        return torch.sum(loss)



