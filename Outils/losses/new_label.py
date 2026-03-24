import torch
import numpy as np


def generate_label(x, x_aug, query, key, batch):
    query = query.squeeze().cpu().detach().numpy()
    x = x.cpu().detach().numpy()   # node_num * 64
    x_aug = x_aug.cpu().detach().numpy()
    w = np.matmul(query.cpu().detach().numpy(), x.T)
    w_aug = np.matmul(query.cpu().detach().numpy(), x_aug.T)

    pass
