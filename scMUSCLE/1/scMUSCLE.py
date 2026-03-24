
from Outils import losses
import math
import sys
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
from DAE_ZINB import DAE_ZINB
from DAE_Ber import DAE_Ber
from torch.optim import Adam
from torch_geometric.utils import dense_to_sparse
# import umap.plot
from matplotlib import pyplot as plt
from scipy.io import loadmat
from process_data import read_dataset, normalize, normalize2
from process_data import dopca
from sklearn.cluster import KMeans
from post_clustering import spectral_clustering, acc, nmi, DI_calcu, JC_calcu
from sklearn import metrics
import scipy.io as sio
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from torch_geometric.nn import GCNConv
import arguments

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)
def binary_cross_entropy(x_pred, x):
    # mask = torch.sign(x)
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
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
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

class AdaptiveGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(AdaptiveGCN, self).__init__()
        self.gcns1 = GCNConv(input_dim, hidden_dim1)
        self.gcns2 = GCNConv(hidden_dim1, hidden_dim2)
        self.gcns3 = GCNConv(hidden_dim2, output_dim)
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim =  output_dim
        self.input_dim = input_dim
        self.dropout = 0.2
        self.lambda_ = 0.8
        self.fc = nn.Linear(hidden_dim1, 16)
    def forward(self, x, edge_index, layers):
        x1 = x
        if layers == 1:
            x = self.gcns1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
            x = self.fc(x)
        elif layers == 2:
            x = self.gcns1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
            x = self.gcns2(x, edge_index)
        elif layers == 3:
            x = self.gcns1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
            x = self.gcns2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
            x = self.gcns3(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        return self.lambda_ * x + (1 - self.lambda_) * x1
    def compute_dist(self, x):
        dist = torch.cdist(x, x, p=2)
        return dist.mean()
def construct_knn_graph(features_X, k=6):
    adj_matrix = kneighbors_graph(features_X, n_neighbors=k, mode='connectivity', include_self=False)
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = csr_matrix(adj_matrix)
    adj_matrix_dense = adj_matrix.toarray()
    return adj_matrix_dense
def Adaptive_probability_graph(features):
    features = normalize_adjacent(features)
    N = np.array(features).shape[0]
    dist1 = compute_dist(np.array(features), np.array(features))
    dist1 = dist1.cpu().numpy() if torch.is_tensor(dist1) else dist1
    dist = dist1 / np.tile(np.sqrt(np.sum(dist1 ** 2, 1))[..., np.newaxis],(1, dist1.shape[1]))
    lam_ = 1
    S = np.zeros((dist.shape[0], dist.shape[1]))
    for k in range(N):
        idxa0 = range(N)
        di = dist[k, :]
        ad = -0.5 * lam_ * di
        S[k][idxa0] = EProjSimplex_new(ad)
    adj = S
    adj = torch.from_numpy(adj)
    binary_adj = torch.where(adj > 0, torch.tensor(1.0), torch.tensor(0.0))
    degree = torch.sum(binary_adj, dim=1)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  # 避免度为 0 的节点导致无穷大
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_normalized = torch.mm(torch.mm(d_mat_inv_sqrt, adj.float()), d_mat_inv_sqrt)
    adj = torch.FloatTensor(adj_normalized)
    edge_index = adj.nonzero(as_tuple=False).T
    return adj,edge_index
def EProjSimplex_new(v, k=1):
    v = np.matrix(v)
    ft = 1
    n = np.max(v.shape)
    if np.min(v.shape) == 0:
        return v, ft
    v0 = v - np.mean(v) + k / n
    vmin = np.min(v0)
    if vmin < 0:
        f = 1
        lambda_m = 0
        while np.abs(f) > 10 ** -10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v1[posidx]) - k
            lambda_m = lambda_m - f / g
            ft = ft + 1
            if ft > 100:
                v1[v1 < 0] = 0.0
                break
        x = v1.copy()
        x[x < 0] = 0.0
    else:
        x = v0
    return x
def compute_dist( array1, array2, type='euclidean'):
    assert type in ['cosine', 'euclidean']
    if isinstance(array1, np.ndarray):
        array1 = torch.from_numpy(array1)
    if isinstance(array2, np.ndarray):
        array2 = torch.from_numpy(array2)
    if torch.cuda.is_available():
        array1 = array1.cuda()
        array2 = array2.cuda()
    if type == 'cosine':
        dist = torch.matmul(array1, array2.T)
        return dist
    else:
        square1 = torch.sum(torch.square(array1), dim=1).unsqueeze(1)
        square2 = torch.sum(torch.square(array2), dim=1).unsqueeze(0)
        t = -2 * torch.matmul(array1, array2.T)
        squared_dist = t + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = torch.sqrt(squared_dist)
        return dist

def normalize_adjacent(nparray, order=2, axis=0):

    nparray = np.array(nparray)
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)
class CentralityEncoding(nn.Module):
    def __init__(self,  node_dim: int,max_degree:int=1047):

        super().__init__()
        self.node_dim = node_dim
        self.max_degree=max_degree

        self.z_degree = nn.Parameter(torch.randn((max_degree+1 , node_dim),requires_grad=True))
        self.linear=nn.Linear(node_dim*2,node_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:

        device = x.device
        self.z_degree = self.z_degree.to(device)
        adj = torch.tensor(adj, device=device)
        adj[adj > 0] = 1
        degree=adj.sum(-1).long()
        degree = self.decrease_to_max_value(degree, self.max_degree)
        degree_embedding=self.z_degree[degree]
        x=torch.cat((0.7*x,0.3*degree_embedding),dim=-1)
        x=self.linear(x)
        return x

    def decrease_to_max_value(self, x, max_value=1047):
        x[x > max_value] = max_value
        return x

def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise
    return p_i_j

def crossview_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))
    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))
    loss = loss.sum()
    return loss

class SelfAttention(nn.Module):
    """
    attention_1
    """
    def __init__(self, dropout):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v):
        queries = q
        keys = k
        values = v
        n, d = queries.shape
        scores = torch.mm(queries, keys.t()) / math.sqrt(d)
        att_weights = F.softmax(scores, dim=1)
        att_emb = torch.mm(self.dropout(att_weights), values)
        return att_weights, att_emb

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 16)

    def forward(self, x, edge_index):
        x1 = x
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2)
        x = self.conv2(x, edge_index)
        λ = 0.2  # gcn的占比
        return λ * x + (1 - λ) * x1

class MLP(nn.Module):
    def __init__(self, z_emb_size1, dropout_rate):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_emb_size1, z_emb_size1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, z_x, z_y):
        q_x = self.mlp(z_x)
        q_y = self.mlp(z_y)
        return q_x, q_y

class Omics_label_Predictor(nn.Module):
    def __init__(self, z_emb_size1):
        super(Omics_label_Predictor, self).__init__()
        self.hidden1 = nn.Linear(z_emb_size1, 5)
        self.hidden2 = nn.Linear(5, 2)
    def forward(self, X):
        X = F.sigmoid(self.hidden1(X))
        y_pre = F.softmax(self.hidden2(X), dim=1)

        return y_pre

class QGrouping(nn.Module):
    def __init__(self, args, in_emb_dim, key_dim=100):
        super(QGrouping, self).__init__()
        embedding_dim = args.embedding_dim
        self.k = args.num_group
        self.key_dim = key_dim
        self.val_dim = embedding_dim // self.k
        args.dim_per_group = embedding_dim // self.k
        self.att_norm = args.att_norm
        self.bias = args.bias

        if self.bias == True:
            self.w_k = torch.nn.Linear(in_emb_dim, self.key_dim, bias=True).to(0)
            self.w_q = torch.nn.Linear(self.key_dim, self.k, bias=True).to(0)
            self.w_v = torch.nn.Linear(in_emb_dim, self.val_dim, bias=True).to(0)
        else:
            self.w_k = torch.nn.Linear(in_emb_dim, self.key_dim, bias=False).to(0)
            self.w_q = torch.nn.Linear(self.key_dim, self.k, bias=False).to(0)
            self.w_v = torch.nn.Linear(in_emb_dim, self.val_dim, bias=False).to(0)

    def forward(self, x):
        key = self.w_k(x)
        val = self.w_v(x)
        que = self.w_q(key)
        embs = []
        norm_w = []
        # x_group = []
        for i in range(x.size(0)):
            if self.att_norm == 'softmax':
                this_w = F.softmax(que[i, :], dim=0).unsqueeze(0)
            elif self.att_norm == 'sigmoid':
                a = torch.sigmoid(que[i, :])
                num_nodes = sum(x.size(0))
                this_w = a / torch.sqrt(num_nodes.float())
            else:
                raise ValueError
            this_val = val[i, :].unsqueeze(0)
            this_embs = torch.matmul(this_w.T, this_val)
            embs.append(this_embs.unsqueeze(0))
            norm_w.append(this_w)
        x_group = torch.cat(embs, dim=0)
        return x_group

class GroupMlp(nn.Module):
    def __init__(self, z_emb_size1):
        super(GroupMlp, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(z_emb_size1 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, z_emb_size1)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(z_emb_size1 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, z_emb_size1)
        )

    def forward(self, z_x_reshape, z_y_reshape):
        z_xy = 0.8* self.mlp1(z_x_reshape) + 0.2 * self.mlp2(z_y_reshape)
        return z_xy

class scMODF(nn.Module):
    def __init__(self, N, in_dim1, in_dim2, hidden1_1, hidden1_2, hidden1_3, z_emb_size1, dropout_rate1,
                 hidden2_1, hidden2_2, hidden2_3, hidden2_4, z_emb_size2, dropout_rate2, params
                 ):
        super(scMODF, self).__init__()
        self.N = N
        self.params = params
        self.DAE_ZINB = DAE_ZINB(in_dim1, hidden1_1, hidden1_2, hidden1_3, z_emb_size1, dropout_rate1)
        self.DAE_Ber = DAE_Ber(in_dim2, hidden2_1, hidden2_2, hidden2_3, hidden2_4, z_emb_size2, dropout_rate2)
        self.attlayer1 = SelfAttention(dropout=0.1)
        self.attlayer2 = SelfAttention(dropout=0.1)
        self.attlayer3 = SelfAttention(dropout=0.1)
        self.olp = Omics_label_Predictor(z_emb_size1)
        self.fc = nn.Linear(96, 16)
        self.mlp = MLP(z_emb_size1, dropout_rate=0.1)
        self. cen_Encoding=CentralityEncoding(z_emb_size1)
        self.Cost_emb, _ = losses.select_loss(self.params.loss_emb, self.params)
        self.Cost_div, _ = losses.select_loss(self.params.loss_div, self.params)
        self.grouping = QGrouping(args=self.params,in_emb_dim=16)
        self.GroupMLP=GroupMlp(z_emb_size1)
    def forward(self, x1, x2, scale_factor):
        z_x = self.DAE_ZINB.fc_encoder(x1).to(device)
        z_y = self.DAE_Ber.fc_encoder(x2).to(device)
        x_group = self.grouping(z_x)
        x_group_a= self.grouping(z_y)
        z_x_reshape = x_group.reshape(1047, -1)
        z_y_reshape = x_group_a.reshape(1047, -1)
        z_xy = torch.cat([z_x_reshape,z_y_reshape],dim=1)
        z_xy=self.fc(z_xy)
        Cost_emb, _ = losses.select_loss(self.params.loss_emb, self.params)
        Cost_div, _ = losses.select_loss(self.params.loss_div, self.params)
        loss_sl_1 = Cost_emb(x_group, x_group_a)
        loss_sl_2 = Cost_div(x_group, x_group  )
        group_loss = loss_sl_1 + self.params.lam_div * loss_sl_2
        features_x = z_x.data.cpu().numpy()
        adjacent_x,edge_index = Adaptive_probability_graph(features_x)
        features_y = z_y.data.cpu().numpy()
        adjacent_y,edge_index = Adaptive_probability_graph(features_y)
        z_cen_x=self.cen_Encoding(z_x,adjacent_x)
        z_cen_y = self.cen_Encoding(z_y,adjacent_y)
        zx_weights, z_gx = self.attlayer1(z_cen_x, z_cen_x, z_cen_x)
        zy_weights, z_gy = self.attlayer2(z_cen_y, z_cen_y, z_cen_y)
        z_conxy = torch.cat([z_gx, z_gy], dim=0)
        y_pre = self.olp(z_conxy)
        q_x, q_y = self.mlp(z_x, z_y)
        z_I =self.params.lam * z_gx + self.params.beta* z_gy + z_xy
        latent_zinb = self.DAE_ZINB.fc_decoder(z_I)
        normalized_x_zinb = F.softmax(self.DAE_ZINB.decoder_scale(latent_zinb), dim=1)
        batch_size = normalized_x_zinb.size(0)
        scale_factor.resize_(batch_size, 1)
        scale_factor.repeat(1, normalized_x_zinb.size(1))
        scale_x_zinb = torch.exp(scale_factor) * normalized_x_zinb
        disper_x_zinb = torch.exp(self.DAE_ZINB.decoder_r(latent_zinb))
        dropout_rate_zinb = self.DAE_ZINB.dropout(latent_zinb)
        latent_ber = self.DAE_Ber.fc_decoder(z_I)
        recon_x_ber = self.DAE_Ber.decoder_scale(latent_ber)
        Final_x_ber = torch.sigmoid(recon_x_ber)
        return z_x, z_y, z_gx, z_gy, q_x, q_y, z_xy, z_I, y_pre,group_loss,\
            normalized_x_zinb, dropout_rate_zinb, disper_x_zinb, scale_x_zinb, Final_x_ber,z_xy

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    torch.cuda.cudnn_enabled = False
    torch.manual_seed(1000)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('===== Using device: ' + device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_path1', type=str,
                        default='dae_scRNA.pkl')
    parser.add_argument('--pretrain_path2', type=str,
                        default='dae_scATAC.pkl')
    parser.add_argument('--pretrain_path3', type=str,
                        default='/data/file/')
    parser.add_argument('--epoch1', type=int, default=500, help='Number of epochs to training_model.')
    parser.add_argument('--training_dae_scRNA', type=bool, default=True, help='Training dae.')
    parser.add_argument('--lam', type=int, default=0.1, help='omics fusion for Z_{gY}')
    parser.add_argument('--beta', type=int, default=1, help='omics fusion for Z_{XY}')
    parser.add_argument('--alpha1', type=int, default=0.0001, help='weight of loss_ber')
    parser.add_argument('--alpha2', type=int, default=1, help='weight of loss_dis')
    parser.add_argument('--alpha3', type=int, default=0.1, help='weight of loss_cl')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--log-interval', default=1, type=int)
    parser.add_argument('--att-norm', type=str, default='softmax')
    parser.add_argument('--add_global_group', action='store_true')
    parser.add_argument('--lam_glb', default=0.01, type=float)
    parser.add_argument('--learning_rate', nargs='*', default=[0.001], type=float)
    parser.add_argument('--weight_decay', nargs='*', default=[0.00] * 5,type=float, help='conv, bn, que, key, val')
    parser.add_argument('--reduction_ratio', nargs='*', default=[1],type=int, help='reduction ratio of key')
    parser.add_argument('--embedding_dim', type=int, default=48)
    parser.add_argument('--aug', nargs=2, default=['dnodes', 'dnodes'], type=str)
    parser.add_argument('--aug_ratio', nargs=2, default=[0.1, 0.1], type=float)
    parser.add_argument('--loss_emb', type=str, default='binomial_deviance')
    parser.add_argument('--loss_div', type=str, default='div_bd')
    parser.add_argument('--lam_div', default=0.5, type=float)
    parser.add_argument('--num_group', default=3,type=int, help='reduction ratio of key')
    parser.add_argument('--top_k', default=11, type=int)
    parser.add_argument('--feat_str', type=str, default='deg+odeg100')
    parser.add_argument('--bias', type=str, default='true')
    parser.add_argument('--club_type', type=str, default='CLUBSample')
    parser.add_argument('--club_fea_norm', type=str, default='false')
    parser.add_argument('--club_learn_mu', type=str, default='true')
    parser.add_argument('--pre_transform', type=str, default='true')
    parser.add_argument('--loss_margin_margin', nargs='*', default=0.1, type=float)
    parser.add_argument('--loss_margin_beta', nargs='*', default=0.8, type=float)
    parser.add_argument('--loss_margin_beta_constant', action='store_true')
    parser.add_argument('--loss_margin_beta_lr', nargs='*', default=0.5, type=float)
    parser.add_argument('--club_hidden', nargs='*', default=4, type=int)
    params = parser.parse_args()
    params.device = device
    print('===== Load scRNA-seq and scATAC data together =====')

    data_root = '/data/'

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

    x2 = normalize(x2, filter_min_counts=True,
                   size_factors=False, normalize_input=False,
                   logtrans_input=False)

    print('===== Normalize =====')

    x_scRNA = x1.X
    x_scRNAraw = x1.raw.X
    x_scRNA_size_factor = x1.obs['size_factors'].values
    x_scRNA = torch.from_numpy(x_scRNA).to(device)
    x_scRNAraw = torch.from_numpy(x_scRNAraw).to(device)
    x_scRNA_size_factor = torch.from_numpy(x_scRNA_size_factor).to(device)
    x_scATAC = x2.X
    x_scATACraw = x2.raw.X
    x_scATAC_size_factor = x2.obs['size_factors'].values

    x_scATAC = torch.from_numpy(x_scATAC).to(device)
    x_scATACraw = torch.from_numpy(x_scATACraw).to(device)
    classes, label_ground_truth = np.unique(label_ground_truth, return_inverse=True)
    classes = classes.tolist()

    N1, M1 = np.shape(x_scRNA)
    N2, M2 = np.shape(x_scATAC)

    ol_x1 = torch.ones(N1, 1)
    ol_x2 = torch.zeros(N1, 1)
    ol_y1 = torch.zeros(N2, 1)
    ol_y2 = torch.ones(N2, 1)
    ol_x = torch.cat([ol_x1, ol_x2], dim=1).to(device)
    ol_y = torch.cat([ol_y1, ol_y2], dim=1).to(device)
    ol = torch.cat([ol_x, ol_y], dim=0).to(device)
    ol1 = ol.cpu().numpy()

    ce_loss = nn.CrossEntropyLoss()
    if params.training_dae_scRNA:
        print("===== Pretrain a scMODF.")
        scMCs = scMODF(N1, M1, M2,
                       hidden1_1=500, hidden1_2=300, hidden1_3=128, z_emb_size1=16, dropout_rate1=0.1,
                       hidden2_1=3000, hidden2_2=2500, hidden2_3=1000, hidden2_4=128, z_emb_size2=16, dropout_rate2=0.1,
                       params=params
                       ).to(device)
        DAE_ZINB_state_dict = torch.load(params.pretrain_path1)
        scMCs.DAE_ZINB.load_state_dict(DAE_ZINB_state_dict)

        DAE_Ber_state_dict = torch.load(params.pretrain_path2)
        scMCs.DAE_Ber.load_state_dict(DAE_Ber_state_dict)
        print("===== Pretrained weights are loaded successfully.")

        optimizer = Adam(scMCs.parameters(), lr=0.0005)
        train_loss_list1 = []
        ans1 = []
        ans2 = []
        for epoch in range(params.epoch1):
            total_loss = 0
            optimizer.zero_grad()

            z_x, z_y, z_gx, z_gy, q_x, q_y, z_xy, z_I, y_pre, cl_loss, \
                normalized_x_zinb, dropout_rate_zinb, disper_x_zinb, scale_x_zinb, Final_x_ber,x_group, x_group_a = scMCs(x_scRNA,
                                                                                                       x_scATAC,
                                                                                                       x_scRNA_size_factor)


            loss_zinb = torch.mean(log_zinb_positive(x_scRNA, scale_x_zinb, disper_x_zinb, dropout_rate_zinb, eps=1e-8))

            loss_Ber = torch.mean(binary_cross_entropy(Final_x_ber, x_scATAC))

            loss_ce = ce_loss(y_pre, ol)
            y_pre = y_pre.detach().cpu().numpy()

            # contrastive loss 对比损失: 直接使用前向传播返回的对比损失 cl_loss。
            loss_cl = cl_loss

            # 组对比损失
            Cost_emb, _ = losses.select_loss(params.loss_emb, params)
            Cost_div, _ = losses.select_loss(params.loss_div,params)

            loss_sl_1 = Cost_emb(x_group, x_group_a)
            loss_sl_2 = Cost_div(x_group, x_group)

            # 总损失: 计算总损失，按权重组合不同损失项。
            loss = loss_zinb + params.alpha1 * loss_Ber + params.alpha2 * loss_ce + params.alpha3 * loss_cl
            loss.backward()
            optimizer.step()

            train_loss_list1.append(loss.item())
            print("epoch {} => loss_zinb={:.4f} loss_Ber={:.4f} loss_ce={:.4f} loss_cl={:.4f} loss={:.4f}".format(epoch,
                                                                                                                  loss_zinb,
                                                                                                                  loss_Ber,
                                                                                                                  loss_ce,
                                                                                                                  loss_cl,
                                                                                                                  loss))

        print("===== save as .mat(txt) and visualization on scRNA-seq")
        z_x = z_gx.data.cpu().numpy()
        z_y = z_gy.data.cpu().numpy()

        z_xxy = z_I.data.cpu().numpy()
        _,adjacent_z_I = Adaptive_probability_graph(z_xxy)

        z_xxy = torch.tensor(z_xxy).float()
        z_xxy = z_xxy.to(device)
        adjacent_z_I = adjacent_z_I.to(device)
        data = Data(x=z_xxy, edge_index=adjacent_z_I)
        model = GCN().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.mse_loss(out, data.x)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
        with torch.no_grad():
            model.eval()
            z_xxy = model(data.x, data.edge_index)
        reducer = umap.UMAP(n_neighbors=15, n_components=2, metric="euclidean")
        z_xxy=z_xxy.data.cpu().numpy()
        z_I_umap = reducer.fit_transform(z_xxy)

        kmeans = KMeans(n_clusters=4, random_state=100)
        label_pred_z_I_umap = kmeans.fit_predict(z_I_umap)

        scatter = plt.scatter(z_I_umap[:, 0], z_I_umap[:, 1], c=label_ground_truth, s=10)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        plt.show()

        NMI_z_I = normalized_mutual_info_score(label_ground_truth, label_pred_z_I_umap, average_method='max')
        NMI_z_I = normalized_mutual_info_score(label_ground_truth, label_pred_z_I_umap, average_method='max')
        ARI_z_I = metrics.adjusted_rand_score(label_ground_truth, label_pred_z_I_umap)
        print('NMI_z_xxy = {:.4f}'.format(NMI_z_I))
        print('ARI_z_xxy = {:.4f}'.format(ARI_z_I))
    print("model saved to {}.".format(params.pretrain_path3 + "scMCs" + "_alpha1_" + str(params.alpha1)
                                      + "_alpha2_" + str(params.alpha2) + "_alpha3_" + str(
        params.alpha3) + "_lambda_" + str(params.lam)
                                      + "_beta_" + str(params.beta) + "_epoch_" + str(params.epoch1) + ".pkl"))
    print('===== Finished of training =====')


