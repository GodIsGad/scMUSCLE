from .margin import Margin
from .div_bd import DivBD
from . import binomial_deviance
from . import div_bd
from . import contrastive

losses = {'DivBD': DivBD,
          'Margin': Margin,
          'contrastive': contrastive,
          'binomial_deviance': binomial_deviance}


def select(loss_name, opt, to_optim=[]):

    if loss_name not in losses:
        raise NotImplementedError('Loss {} not implemented!'.format(loss_name))

    loss = losses[loss_name](opt)

    if loss.REQUIRES_OPTIM:
        if hasattr(loss, 'optim_dict_list') and loss.optim_dict_list is not None:
            to_optim += loss.optim_dict_list
        else:
            to_optim += [{'params': loss.parameters(), 'lr': loss.lr}]

    return loss, to_optim
