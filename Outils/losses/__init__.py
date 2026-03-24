from . import binomial_deviance, margin, soft_binomial_deviance
from . import negative_log_likelihood, contrastive_loss
from . import normalized_binomial_deviance
from . import div_bd, div_club, div_club_bd, div_KL
from . import dro_loss, binomial_deviance_topk


def select_loss(loss_type, args, to_optim=[]):
    if loss_type == 'margin':
        loss = margin.MarginLoss(args)
        to_optim += [{'params': loss.parameters(), 'lr': args.loss_margin_beta_lr}]
        return loss, to_optim
    elif loss_type == 'binomial_deviance':
        loss = binomial_deviance.BinomialDevianceLoss(args)
        return loss, to_optim
    elif loss_type == 'IG_binomial_deviance':
        loss = binomial_deviance.InfoGraphBDLoss(args)
        return loss, to_optim

    elif loss_type == 'BD_loss':
        loss = binomial_deviance.BDLoss(args)
        return loss, to_optim

    elif loss_type == 'weighted_binomial_deviance':
        loss = binomial_deviance.WeightedBDLoss(args)
        return loss, to_optim

    elif loss_type == 'soft_binomial_deviance':
        loss = soft_binomial_deviance.SoftBinomialDevianceLoss(args)
        to_optim += [{'params': loss.parameters(), 'lr': args.loss_soft_beta_lr}]
        return loss, to_optim
    elif loss_type == 'normalized_binomial_deviance':
        loss = normalized_binomial_deviance.NormalizedBinomialDevianceLoss(args)
        to_optim += [{'params': loss.parameters(), 'lr': args.loss_soft_beta_lr}]
        return loss, to_optim
    elif 'nll' in loss_type:
        loss = negative_log_likelihood.NegativeLogLikelihood(args)
        return loss, to_optim
    elif 'contrastive' in loss_type:
        loss = contrastive_loss.ContrastiveLoss(args)
        return loss
    elif loss_type == 'div_bd':
        loss = div_bd.DivBD(args)
        return loss, to_optim
    elif loss_type == 'div_club':
        loss = div_club.DivCLUB(args)
        return loss, to_optim
    elif loss_type == 'div_club_bd':
        loss = div_club_bd.DivClubBD(args)
        return loss, to_optim
    elif loss_type == 'div_KL':
        loss = div_KL.DivKL(args)
        return loss, to_optim
    elif loss_type == 'dro_topk':
        loss = dro_loss.DroLoss(args)
        return loss, to_optim
    elif loss_type == 'binomial_deviance_topk':
        loss = binomial_deviance_topk.BinomialDevianceTopk(args)
        return loss, to_optim
    else:
        raise ValueError('please input the correct loss type')

