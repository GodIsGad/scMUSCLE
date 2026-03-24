import torch
from Outils import batchminer as bmine
# import batchminer as bmine

ALLOWED_MINING_OPS = list(bmine.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM = True


class Margin(torch.nn.Module):
    def __init__(self, opt):
        super(Margin, self).__init__()

        self.margin = opt.loss_margin_margin
        self.nu = opt.loss_margin_nu
        self.beta_constant = opt.loss_margin_beta_constant

        if self.beta_constant:
            self.beta = opt.loss_margin_beta
        else:
            self.beta = torch.nn.Parameter(torch.ones(opt.k, opt.n_classes) * opt.loss_margin_beta, requires_grad=True)

        if REQUIRES_BATCHMINER:
            self.batchminer = bmine.select(opt)

        self.lr = opt.loss_margin_beta_lr

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def forward(self, batch, labels, **kwargs):
        loss_emb = torch.tensor(0.0).cuda()
        try:
            n, k, dim = batch.shape
        except:
            batch = batch.unsqueeze(1)
            n, k, dim = batch.shape

        for i in range(k):
            this_batch = batch[:, i, :]
            sampled_triplets = self.batchminer(this_batch, labels)
            if self.beta_constant:
                beta = self.beta[i]
            else:
                beta = torch.stack([self.beta[i][labels[triplet[0]]] for triplet in sampled_triplets]).to(
                    torch.float).to('cuda')
            loss_emb += self.margin_loss(this_batch, sampled_triplets, beta)
        loss_emb /= torch.tensor(k, dtype=torch.float32)

        return loss_emb

    def margin_loss(self, batch, sampled_triplets, beta):
        if len(sampled_triplets):
            d_ap, d_an = [], []
            for triplet in sampled_triplets:
                train_triplet = {'Anchor': batch[triplet[0], :], 'Positive': batch[triplet[1], :],
                                 'Negative': batch[triplet[2]]}

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
            loss = torch.tensor(0.).to(torch.float).to(batch.device)

        return loss
