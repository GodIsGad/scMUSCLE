import math
import argparse
import os
import warnings
import seaborn as sns
import anndata
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap.umap_ as umap
from sklearn.decomposition import PCA
from DAE_ZINB import DAE_ZINB
from DAE_Ber import DAE_Ber
from torch.optim import Adam
from sklearn import metrics
import scipy.io as sio
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch.autograd import Variable
from torch.distributions import Normal, kl_divergence as kl
import scanpy as sc
import torch.utils.data as data_utils

# import umap.plot
from matplotlib import pyplot as plt
from scipy.io import loadmat
from process_data import read_dataset, normalize, normalize2
from process_data import dopca
from sklearn.cluster import KMeans
from post_clustering import spectral_clustering, acc, nmi, DI_calcu, JC_calcu
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)
def binary_cross_entropy(x_pred, x):
    #mask = torch.sign(x)
    return - torch.sum(x * torch.log(x_pred + 1e-8) + (1 - x) * torch.log(1 - x_pred + 1e-8), dim=1)
def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    # x = x.to_dense()
    loss_rec = loss_func(decoded, x)
    return loss_rec
def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))
    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log( theta + eps )
    log_theta_mu_eps = torch.log( theta + mu + eps )
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)
    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)
    case_non_zero = (-softplus_pi + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1))
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)
    res = mul_case_zero + mul_case_non_zero
    result = - torch.sum(res, dim=1)
    result = _nan2inf(result)
    return result
class Eucli_dis(nn.Module):
    """like what you like: knowledge distill via neuron selectivity transfer"""
    def __init__(self):
        super(Eucli_dis, self).__init__()
        pass
    def forward(self, g_s, g_t):
        g_s = g_s.float()
        g_t = g_t.float()
        ret = torch.pow( (g_s - g_t) , 2)
        return torch.sum( ret, dim = 1 )
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    torch.cuda.cudnn_enabled = False
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print('===== Using device: ' + device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch1', type=int, default=500, help='Number of epochs to training_dae_scRNA.')
    parser.add_argument('--training_dae_scRNA', type=bool, default=True, help='Training dae.')
    parser.add_argument('--pretrain_path1', type=str,
                        default='dae_scRNA.pkl')
    params = parser.parse_args()
    params.device = device

    print('===== Load scRNA-seq and scATAC data together =====')
    data_root = '/data'
    X1 = os.path.join(data_root, 'scRNA_seq_SNARE.tsv')
    X2 = os.path.join(data_root, 'scATAC_seq_SNARE.txt')
    x3 = os.path.join(data_root, 'cell_metadata.txt')
    x1, x2, train_index, test_index, label_ground_truth, _ = read_dataset(File1=X1, File2=X2,
                                                                          File3=x3, File4=None,
                                                                          transpose=True, test_size_prop=0.0,
                                                                          state=1, format_rna="table",
                                                                          formar_epi="table")
    x1 = normalize(x1, filter_min_counts=True,
                   size_factors=True, normalize_input=False,
                   logtrans_input=True)

    print('===== Normalize =====')

    x_scRNA = x1.X

    x_scRNAraw = x1.raw.X
    x_scRNA_size_factor = x1.obs['size_factors'].values

    x_scRNA = torch.from_numpy(x_scRNA).to(device)
    x_scRNAraw = torch.from_numpy(x_scRNAraw).to(device)
    x_scRNA_size_factor = torch.from_numpy(x_scRNA_size_factor).to(device)

    N1, M1 = np.shape(x_scRNA)

    classes, label_ground_truth = np.unique(label_ground_truth, return_inverse=True)
    classes = classes.tolist()

    if params.training_dae_scRNA:
        print("===== Pretrain a DAE_scRNA.")
        dae_scRNA = DAE_ZINB(M1, hidden1=500, hidden2=300, hidden3=128, z_emb_size=16, dropout_rate=0.1)
        dae_scRNA.to(device)
        print(dae_scRNA)
        optimizer = Adam(dae_scRNA.parameters(), lr=0.0001)
        train_loss_list2 = []

        for epoch in range(params.epoch1):
            total_loss = 0
            optimizer.zero_grad()
            # 前向传播
            emb_scRNA, normalized_scRNA, pi_scRNA, disp_scRNA, mean_scRNA  = dae_scRNA(x_scRNA, x_scRNA_size_factor)
            # 加入knn构图

            loss = torch.mean(log_zinb_positive(x_scRNAraw, mean_scRNA, disp_scRNA, pi_scRNA))
            loss.backward()
            optimizer.step()
            train_loss_list2.append(loss.item())
            print("epoch {} loss={:.4f}".format(epoch, loss))

        emb_scRNA = emb_scRNA.data.cpu().numpy()
        np.savetxt('emb_scRNA.txt', emb_scRNA)
        sio.savemat('emb_scRNA.mat', {'emb_scRNA': emb_scRNA})

        reducer = umap.UMAP(n_neighbors=15, n_components=2, metric="euclidean")
        emb_scRNA_umap = reducer.fit_transform(emb_scRNA)

        np.savetxt('emb_scRNA_umap.txt', emb_scRNA_umap)
        sio.savemat('emb_scRNA_umap.mat', {'emb_scRNA_umap': emb_scRNA_umap})

        kmeans = KMeans(n_clusters=4)
        label_pred_emb_scRNA_umap = kmeans.fit_predict(emb_scRNA_umap)

        colors = np.array([
            [0, 255, 255],
            [128, 0, 255],
            [128, 255, 0],
            [255, 0, 0]
        ]) / 255.0

        point_colors = colors[label_pred_emb_scRNA_umap]

        plt.scatter(emb_scRNA_umap[:, 0], emb_scRNA_umap[:, 1], c=point_colors)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.show()
        SI_dae_scRNA = silhouette_score(emb_scRNA, label_pred_emb_scRNA_umap)
        NMI_dae_scRNA = round(
            normalized_mutual_info_score(label_ground_truth, label_pred_emb_scRNA_umap, average_method='max'),
            3)
        ARI_dae_scRNA = round(metrics.adjusted_rand_score(label_ground_truth, label_pred_emb_scRNA_umap), 3)
        print('NMI_dae_scRNA = {:.4f}'.format(NMI_dae_scRNA))
        print('ARI_dae_scRNA = {:.4f}'.format(ARI_dae_scRNA))
        print('SI_dae_scRNA = {:.4f}'.format(SI_dae_scRNA))
        torch.save(dae_scRNA.state_dict(), params.pretrain_path1)
    print("model saved to {}.".format(params.pretrain_path1))
    print('===== Finished of training_dae_scRNA =====')





