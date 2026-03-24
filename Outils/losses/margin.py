import torch
import torch.nn.functional as F
import sys
sys.path.append('/data/xxy/my-DML-projects/ReML-icml20')
sys.path.append('/data/xxy/GraphCL-master/Outils/losses')
from Outils.criteria import binomial_deviance as bd
from Outils.losses.batch_miner import BatchMiner


class MarginLoss(torch.nn.Module):
    def __init__(self, args):
        super(MarginLoss, self).__init__()
        self.batchminer = BatchMiner(args)
        self.margin = args.loss_margin_margin
        self.k = args.num_group
        self.lam_div = args.lam_div

        if args.loss_margin_beta_constant:
            self.beta = args.loss_margin_beta
        else:
            self.beta = torch.nn.Parameter(torch.ones(self.k, args.n_classes) * args.loss_margin_beta)

    def forward(self, xs, x_augs, *vars):
        ind = vars[0]
        xs = F.normalize(xs, p=2, dim=-1)
        x_augs = F.normalize(x_augs, p=2, dim=-1)
        loss_emb, loss_global, loss_div = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        _, k, _ = xs.size()
        y2 = torch.cat([torch.arange(xs.shape[0]), torch.arange(xs.shape[0])], dim=0)
        ind2 = torch.cat([ind, ind], dim=0)
        for i in range(k):
            x, x_aug = xs[:, i, :], x_augs[:, i, :]
            x2 = torch.cat([x, x_aug], dim=0)
            sampled_triplets = self.batchminer(x2, y2)

            beta = torch.stack([self.beta[i][ind2[triplet[0]]] for triplet in sampled_triplets]).to(
                torch.float).to('cuda')
            loss_emb += self.margin_loss(x2, sampled_triplets, beta)

        loss_emb /= torch.tensor(k, dtype=torch.float32)
        loss_div += bd.dirdiv_bd(xs)
        loss_div += bd.dirdiv_bd(x_augs)
        loss_div /= 2

        loss = loss_emb + self.lam_div * loss_div

        return loss, loss_emb, loss_div, loss_global

    def margin_loss(self, x, sampled_triplets, beta):
        if len(sampled_triplets):
            d_ap, d_an = [], []
            for triplet in sampled_triplets:
                train_triplet = {'Anchor': x[triplet[0], :], 'Positive': x[triplet[1], :],
                                 'Negative': x[triplet[2]]}

                pos_dist = ((train_triplet['Anchor'] - train_triplet['Positive']).pow(2).sum() + 1e-8).pow(1 / 2)
                neg_dist = ((train_triplet['Anchor'] - train_triplet['Negative']).pow(2).sum() + 1e-8).pow(1 / 2)

                d_ap.append(pos_dist)
                d_an.append(neg_dist)
            d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

            pos_loss = torch.nn.functional.relu(d_ap - beta + self.margin)
            neg_loss = torch.nn.functional.relu(beta - d_an + self.margin)

            pair_count = torch.sum((pos_loss > 0.) + (neg_loss > 0.)).to(torch.float).to(d_ap.device)

            if pair_count == 0.:
                loss = torch.sum(pos_loss + neg_loss)
            else:
                loss = torch.sum(pos_loss + neg_loss) / pair_count

        else:
            loss = torch.tensor(0.).to(torch.float).to(x.device)

        return loss
