import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import Outils.batchminer as batchminer
# from Outils.criteria import binomial_deviance


"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False


class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        super(Criterion, self).__init__()
        self.pos_margin = opt.loss_contrastive_pos_margin
        self.neg_margin = opt.loss_contrastive_neg_margin
        self.batchminer = batchminer

        self.name           = 'contrastive'
        self.group_name = opt.group_name
        self.opt = opt

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def contrastive(self, batch, labels):
        sampled_triplets = self.batchminer(batch, labels)

        anchors = [triplet[0] for triplet in sampled_triplets]
        positives = [triplet[1] for triplet in sampled_triplets]
        negatives = [triplet[2] for triplet in sampled_triplets]
        pos_dists = torch.mean(
            F.relu(nn.PairwiseDistance(p=2)(batch[anchors, :], batch[positives, :]) - self.pos_margin))
        neg_dists = torch.mean(
            F.relu(self.neg_margin - nn.PairwiseDistance(p=2)(batch[anchors, :], batch[negatives, :])))

        loss = pos_dists + neg_dists
        return loss

    def forward(self, batch, labels, **kwargs):
        batch = F.normalize(batch, dim=-1)
        loss_emb, loss_div = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        if self.group_name != '':
            n, k, dim = batch.size()
            for i in range(k):
                this_batch = batch[:, i, :]
                loss_emb += self.contrastive(this_batch, labels)
            loss_emb /= torch.tensor(k, dtype=torch.float32)
            loss_div += eval('diversity.{}(batch, self.opt)'.format(self.opt.diversity))
        else:
            loss_emb = self.contrastive(batch, labels)

        loss = self.opt.lam_emb * loss_emb + self.opt.lam_div * loss_div
        return loss, loss_emb, loss_div



    # def forward(self, batch, labels, **kwargs):
    #     batch = F.normalize(batch, dim=-1)
    #     sampled_triplets = self.batchminer(batch, labels)
    #
    #     anchors   = [triplet[0] for triplet in sampled_triplets]
    #     positives = [triplet[1] for triplet in sampled_triplets]
    #     negatives = [triplet[2] for triplet in sampled_triplets]
    #
    #     pos_dists = torch.mean(F.relu(nn.PairwiseDistance(p=2)(batch[anchors,:], batch[positives,:]) -  self.pos_margin))
    #     neg_dists = torch.mean(F.relu(self.neg_margin - nn.PairwiseDistance(p=2)(batch[anchors,:], batch[negatives,:])))
    #
    #     loss      = pos_dists + neg_dists
    #
    #     return loss
