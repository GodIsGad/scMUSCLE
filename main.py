_author_ = 'lrren'
# coding: utf-8

# training_scmifc
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
# 这段代码实现了一个多组学数据融合模型 scMODF 的训练，包括单细胞RNA测序（scRNA-seq）和单细胞ATAC测序（scATAC-seq）数据。以下是对这段代码的详细解释。
from scipy.sparse import csr_matrix
from torch_geometric.nn import GCNConv
import arguments
# 这个函数用于将张量中的 NaN（Not a Number）值替换为无穷大（Inf）。具体实现步骤如下：
# torch.isnan(x) 检查张量 x 中的每个元素是否为 NaN。
# 使用 torch.where 替换 NaN 值：如果元素为 NaN，则替换为无穷大（torch.zeros_like(x) + np.inf），否则保持原值不变。
def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


# 这个函数计算二值交叉熵损失，用于衡量两个概率分布之间的差异，通常用于二分类问题。具体实现步骤如下：
# 对预测值 x_pred 和真实值 x 逐元素计算交叉熵。
# 使用 torch.log 计算对数，并添加一个很小的值 1e-8 以防止对数零的情况。
# 对结果进行求和以获得损失值。
def binary_cross_entropy(x_pred, x):
    # mask = torch.sign(x)
    return - torch.sum(x * torch.log(x_pred + 1e-8) + (1 - x) * torch.log(1 - x_pred + 1e-8), dim=1)


# 这个函数计算重建误差，使用均方误差（Mean Squared Error, MSE）作为损失函数。具体实现步骤如下：
# 定义 MSE 损失函数 torch.nn.MSELoss()。
# 计算重建张量 decoded 和原始张量 x 之间的 MSE 损失。
def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    # x = x.to_dense()
    loss_rec = loss_func(decoded, x)
    return loss_rec


# 这个函数计算零膨胀负二项分布（ZINB）模型的对数似然。具体实现步骤如下：
# 如果 theta 是一维的，将其变为二维。
# 计算 pi 的 softplus，theta 和 mu 的对数值。
# 计算 pi 和 theta 的对数概率值。
# 计算两种情况下的对数似然值：观测值为零和非零。
# 根据观测值的情况选择对应的对数似然值。
# 求和得到最终的对数似然值，并将 NaN 值替换为无穷大。
def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    # x = x.float()

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

# knn构图
def construct_knn_graph(features_X, k=6):  #

    adj_matrix = kneighbors_graph(features_X, n_neighbors=k, mode='connectivity', include_self=False)
    # 输入数据矩阵，形状为 (n_samples, n_features)。n_samples 是样本数量，n_features 是特征数量。
    # connectivity'：返回二值邻接矩阵，其中1表示连接，0表示没有连接。
    # include_self: 是否在邻接矩阵中包含样本点自身的连接。
    # 确保邻接矩阵是稀疏矩阵
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = csr_matrix(adj_matrix)
    adj_matrix_dense = adj_matrix.toarray() # 转化为稠密矩阵

    # 　  features = z_x.data.cpu().numpy()
    #  A_zx=construct_knn_graph(features)
   # adj_matrix = normalize_adj(adj_matrix)  # 对邻接矩阵进行归一化
 #  edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
    return adj_matrix_dense
def Adaptive_probability_graph(features):  # th 很重要
    features = normalize_adjacent(features)
    N = np.array(features).shape[0]

    dist1 = compute_dist(np.array(features), np.array(features))
    dist1 = dist1.cpu().numpy() if torch.is_tensor(dist1) else dist1
    #max_dist1 = np.max(dist1, axis=0)
    dist = dist1 / np.tile(np.sqrt(np.sum(dist1 ** 2, 1))[..., np.newaxis],(1, dist1.shape[1]))  # normalization for distance


    lam_ = 1  # 很重要，也可以调解
    S = np.zeros((dist.shape[0], dist.shape[1]))
    for k in range(N):
        idxa0 = range(N)
        di = dist[k, :]
        ad = -0.5 * lam_ * di
        S[k][idxa0] = EProjSimplex_new(ad)

    adj = S  # Adjacency matrix
    adj = torch.from_numpy(adj)
    # 增加自环，即 A' = A + I
    binary_adj = torch.where(adj > 0, torch.tensor(1.0), torch.tensor(0.0))

    # 计算度矩阵 D，并取对角线元素
    degree = torch.sum(binary_adj, dim=1)

    # 计算 D^(-1/2)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  # 避免度为 0 的节点导致无穷大

    # 构建 D^(-1/2) 矩阵
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

    # 计算标准化后的邻接矩阵
    adj_normalized = torch.mm(torch.mm(d_mat_inv_sqrt, adj.float()), d_mat_inv_sqrt)

    adj = torch.FloatTensor(adj_normalized)
  #  S_dist = torch.FloatTensor(S_dist)
    # 将邻接矩阵转化为边的索引
   # edge_index = adj.nonzero(as_tuple=False).T
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
    """Compute the euclidean or cosine distance of all pairs.
    Args:
        array1: numpy array or torch tensor on GPU with shape [m1, n]
        array2: numpy array or torch tensor on GPU with shape [m2, n]
    #    type: one of ['cosine', 'euclidean']
    Returns:
        numpy array or torch tensor on GPU with shape [m1, m2]
    """

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
    """Normalize a N-D numpy array along the specified axis."""
    nparray = np.array(nparray)
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)
class CentralityEncoding(nn.Module):
    def __init__(self,  node_dim: int,max_degree:int=1047):
        # max_degree：原代码中是对出入度进行限制，这里直接对总的节点的最大度数限制，这里设置为5。
        # node_dim：节点特征的维度。
        #
        super().__init__()
        self.node_dim = node_dim
        self.max_degree=max_degree

        # 创建一个参数矩阵，用于存储每个度数对应的嵌入向量。使用 max_degree + 1 来包括度数为0的情况。
        # torch.randn((max_degree + 1, node_dim)) 生成一个大小为 (max_degree + 1, node_dim) 的随机张量。
        self.z_degree = nn.Parameter(torch.randn((max_degree+1 , node_dim),requires_grad=True))
        self.linear=nn.Linear(node_dim*2,node_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:# 输入 特征x   输入一个adj

        device = x.device  # 获取输入张量所在的设备
        self.z_degree = self.z_degree.to(device)  # 将度嵌入矩阵移到同一设备
        # 将邻接矩阵从 numpy 数组转换为 PyTorch 张量，并移动到同一设备
        adj = torch.tensor(adj, device=device)
        adj[adj > 0] = 1
        degree=adj.sum(-1).long() # 计算每个节点的度数。
        degree = self.decrease_to_max_value(degree, self.max_degree) # 将度数限制在最大值 self.max_degree 范围内。
        degree_embedding=self.z_degree[degree]
        x=torch.cat((0.7*x,0.3*degree_embedding),dim=-1)
        x=self.linear(x)
        return x # 返回特征矩阵

    def decrease_to_max_value(self, x, max_value=1047):
        "限制节点度的最大值"
        x[x > max_value] = max_value
        return x
# 对比损失函数和自注意力模块
# contrastive loss
# 这个函数用于计算联合概率矩阵 P。具体实现步骤如下：
# 获取 view1 和 view2 的尺寸。
# 断言 view1 和 view2 的尺寸匹配。
# 计算联合概率矩阵 p_i_j：
# 使用 unsqueeze 扩展维度，然后逐元素相乘。
# 沿第0维求和。
# 对称化（取平均）。
# 归一化（除以总和）。
def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


# 对比损失函数用于最大化不同视图之间的相似性
# 自注意力模块用于捕捉输入之间的相互关系，
# MLP 则用于将输入投影到相同的特征空间。这些组件在多组学数据融合和分析中起到了关键作用。


# 这个函数用于计算视图之间的对比损失，以最大化一致性。具体实现步骤如下：
# 获取 view1 的尺寸。
# 计算联合概率矩阵 p_i_j。
# 断言 p_i_j 的尺寸正确。
# 计算边缘概率 p_i 和 p_j。
# 使用 torch.where 将小于 EPS 的值替换为 EPS。
# 计算对比损失：
# 对每个元素计算对比损失。
# 求和得到总损失。
def crossview_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)


    # Works with pytorch > 1.2
    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()

    return loss


# 自注意力模块
class SelfAttention(nn.Module):
    """
    attention_1
    """

    def __init__(self, dropout):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # 获取查询、键和值。
        queries = q
        keys = k
        values = v
        n, d = queries.shape  # 获取查询的批次大小和特征维度
        scores = torch.mm(queries, keys.t()) / math.sqrt(d)  # 计算查询和键的点积，并除以特征维度的平方根进行缩放。
        att_weights = F.softmax(scores, dim=1)  # 对分数进行 softmax 操作，得到注意力权重。
        att_emb = torch.mm(self.dropout(att_weights), values)  # 使用注意力权重加权求和值得到注意力嵌入。
        return att_weights, att_emb  # 返回注意力权重和注意力嵌入。

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
        #return  x

# 多层感知机  实现对比学习
# 实现一个简单的多层感知机（MLP），用于将输入投影到相同的空间。
class MLP(nn.Module):

    def __init__(self, z_emb_size1, dropout_rate):
        super(MLP, self).__init__()
        # 定义 MLP 结构，包括线性层、ReLU 激活函数和 Dropout 层。
        self.mlp = nn.Sequential(
            nn.Linear(z_emb_size1, z_emb_size1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # nn.Linear(z_emb_size1, z_emb_size1),
            # nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
        )

    def forward(self, z_x, z_y):
        q_x = self.mlp(z_x)  # 将输入 z_x 投影到新的空间
        q_y = self.mlp(z_y)  # 将输入 z_y 投影到新的空间。
        return q_x, q_y


# 组学标签预测器由2层全连接层实现
class Omics_label_Predictor(nn.Module):
    def __init__(self, z_emb_size1):
        super(Omics_label_Predictor, self).__init__()

        # input to first hidden layer
        self.hidden1 = nn.Linear(z_emb_size1, 5)

        # second hidden layer and output
        self.hidden2 = nn.Linear(5, 2)

    def forward(self, X):
        X = F.sigmoid(self.hidden1(X))
        y_pre = F.softmax(self.hidden2(X), dim=1)
        # y_pre = F.sigmoid(self.hidden2(X))
        return y_pre


class QGrouping(nn.Module):
    def __init__(self, args, in_emb_dim, key_dim=100):
        super(QGrouping, self).__init__()
        # cprint(f'Initail a query grouping layer', 'green')
        embedding_dim = args.embedding_dim  # 16*组数

        self.k = args.num_group  # k=3

        self.key_dim = key_dim  # 100
        self.val_dim = embedding_dim // self.k  # 16
        args.dim_per_group = embedding_dim // self.k  # 64
        self.att_norm = args.att_norm  # softmax
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
        # x = x.unsqueeze(0)  # 补齐batch_size
        key = self.w_k(x)
        val = self.w_v(x)
        que = self.w_q(key)
        # weight = softmax(each(Q),K^T)
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
        # 使用 nn.Sequential 定义多层感知机
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
        # 正确调用 self.mlp1 和 self.mlp2
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
        # 调用中心性编码
        self. cen_Encoding=CentralityEncoding(z_emb_size1)
        # 加入组对比参数
        self.Cost_emb, _ = losses.select_loss(self.params.loss_emb, self.params)
        self.Cost_div, _ = losses.select_loss(self.params.loss_div, self.params)
        self.grouping = QGrouping(args=self.params,in_emb_dim=16) # embedding_dim*3 16*3
        self.GroupMLP=GroupMlp(z_emb_size1)
    def forward(self, x1, x2, scale_factor):  # scale_factor有什么用
        ## encoder
        # 得到编码后的表示
        z_x = self.DAE_ZINB.fc_encoder(x1).to(device)
        z_y = self.DAE_Ber.fc_encoder(x2).to(device)

        # 加入组对比
        x_group = self.grouping(z_x)
        x_group_a= self.grouping(z_y)
        # 对RNA特征扩增后：(1047，3，16）变为(1047，16），有3种操作，1.取平均，2.第二维度求和，3.reshape重塑性状,4.取第二个维度的某一个特征
        z_x_reshape = x_group.reshape(1047, -1)
        z_y_reshape = x_group_a.reshape(1047, -1)
        z_xy = torch.cat([z_x_reshape,z_y_reshape],dim=1)
        z_xy=self.fc(z_xy)


        Cost_emb, _ = losses.select_loss(self.params.loss_emb, self.params)
        Cost_div, _ = losses.select_loss(self.params.loss_div, self.params)
        loss_sl_1 = Cost_emb(x_group, x_group_a)
        loss_sl_2 = Cost_div(x_group, x_group  )
        group_loss = loss_sl_1 + self.params.lam_div * loss_sl_2

        # 加入中心性编码，先构造邻接矩阵
        features_x = z_x.data.cpu().numpy()
        adjacent_x,edge_index = Adaptive_probability_graph(features_x)

        features_y = z_y.data.cpu().numpy()
        adjacent_y,edge_index = Adaptive_probability_graph(features_y)


        z_cen_x=self.cen_Encoding(z_x,adjacent_x)

        z_cen_y = self.cen_Encoding(z_y,adjacent_y)

        # 得到注意力权重，注意力嵌入  z_gx=Ax*z_x
        ## attention for omics specific information of scRNA-seq
        zx_weights, z_gx = self.attlayer1(z_cen_x, z_cen_x, z_cen_x)

        ## attention for omics specific information of scATAC
        zy_weights, z_gy = self.attlayer2(z_cen_y, z_cen_y, z_cen_y)




        # 将得到的注意力嵌入连接起来形成z_conxy
        # # omics-label predictor 组学标签预测器
        z_conxy = torch.cat([z_gx, z_gy], dim=0)

        # 　z_conxy输入到预测器中得到预测的标签y_pre
        y_pre = self.olp(z_conxy)

        # omics-label predictor
        # z_conxy = torch.cat([z_x, z_y], dim=0)
        # y_pre_zx = self.olp(z_x)
        # y_pre_zy = self.olp(z_y)

        # 将 z_x 和 z_y 投影到相同的空间，得到 q_x 和 q_y。
        ## cell similarity cross scRNA and scATAC
        # project z_x and z_y into the same space 将编码后的emb得到q，然后对比学习
        q_x, q_y = self.mlp(z_x, z_y)

        # contrastive loss to maximize consistency　计算对比损失
     #   cl_loss = crossview_contrastive_Loss(q_x, q_y)

        # capture the consistency information　将 q_x 和 q_y 连接起来，形成 emb_con。　将 emb_con 输入到全连接层，得到 z_xy。
       # emb_con = torch.cat([q_x, q_y], dim=1)
      #  z_xy = self.fc(emb_con)
        # z_xy = (q_x + q_y)/2

        # z_I = z_gx + self.params.lam * z_gy + self.params.beta * z_xy　　整合来自 scRNA-seq 和 scATAC-seq 的信息，形成最终的综合表示 z_I。
        z_I = self.params.beta * z_gx + self.params.lam * z_gy + z_xy


        # decoder for DAE_ZINB　　将 z_I 输入到 DAE_ZINB 的解码器，得到解码后的潜在表示 latent_zinb。
        latent_zinb = self.DAE_ZINB.fc_decoder(z_I)

        # 　应用 Softmax 激活，得到标准化的重建数据 normalized_x_zinb
        normalized_x_zinb = F.softmax(self.DAE_ZINB.decoder_scale(latent_zinb), dim=1)
        batch_size = normalized_x_zinb.size(0)
        # 　调整 scale_factor 的尺寸。
        scale_factor.resize_(batch_size, 1)
        # 　扩展 scale_factor 的维度以匹配 normalized_x_zinb。
        scale_factor.repeat(1, normalized_x_zinb.size(1))

        scale_x_zinb = torch.exp(scale_factor) * normalized_x_zinb  # recon_x
        # scale_x = normalized_x  # recon_x

        # 计算分布参数 disper_x_zinb和dropout率
        disper_x_zinb = torch.exp(self.DAE_ZINB.decoder_r(latent_zinb))  # theta
        dropout_rate_zinb = self.DAE_ZINB.dropout(latent_zinb)  # pi

        # decoder for DAE_Ber
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

    # ################ Parameter setting
    parser = argparse.ArgumentParser()
    # 加载训练的pkl数据，放入路径，先训练scRNA,scATAC
    # 预训练路径: 添加三个参数 --pretrain_path1、--pretrain_path2 和 --pretrain_path3 分别指定预训练模型和保存模型的路径。
    parser.add_argument('--pretrain_path1', type=str,
                        default='/data/jyz/scMCs/pkl_baseLine_0.89_0.92/dae_scRNA.pkl')
    parser.add_argument('--pretrain_path2', type=str,
                        default='/data/jyz/scMCs/pkl_baseLine_0.89_0.92/dae_scATAC.pkl')
    # 保存加载的数据
    parser.add_argument('--pretrain_path3', type=str,
                        default='/data/jyz/scMCs/file/')

    parser.add_argument('--epoch1', type=int, default=500, help='Number of epochs to training_model.')
    parser.add_argument('--training_dae_scRNA', type=bool, default=True, help='Training dae.')
    # parameters for multi-omics data fusion
    # 多组学数据融合参数: 添加参数 --lam 和 --beta 分别指定组学融合的权重。
    parser.add_argument('--lam', type=int, default=0.1, help='omics fusion for Z_{gY}')
    parser.add_argument('--beta', type=int, default=1, help='omics fusion for Z_{XY}')
    # parameters for model optimization
    # 模型优化参数: 添加参数 --alpha1、--alpha2 和 --alpha3 分别指定不同损失项的权重。
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
    # parser.add_argument('--loss_margin_beta', nargs='*', default=[0.01, 0.02, 0.03, 0.04, 0.05], type=float)
    parser.add_argument('--loss_margin_beta', nargs='*', default=0.8, type=float)

    parser.add_argument('--loss_margin_beta_constant', action='store_true')
    parser.add_argument('--loss_margin_beta_lr', nargs='*', default=0.5, type=float)
    parser.add_argument('--club_hidden', nargs='*', default=4, type=int)

    params = parser.parse_args()
    params.device = device

    # ======================================================================= read data from the data_root folder
    print('===== Load scRNA-seq and scATAC data together =====')

    data_root = '/data/jyz/dataset/scMCs/'

    X1 = os.path.join(data_root, 'scRNA_seq_SNARE.tsv')
    X2 = os.path.join(data_root, 'scATAC_seq_SNARE.txt')
    x3 = os.path.join(data_root, 'cell_metadata.txt')  # cell type information
    print("==================数据加载完成==================")
    # # adata: scRNA-seq with samples x genes
    # # adata: scATAC    with samples x peaks
    x1, x2, train_index, test_index, label_ground_truth, _ = read_dataset(File1=X1, File2=X2,
                                                                          File3=x3, File4=None,
                                                                          transpose=True, test_size_prop=0.0,
                                                                          state=1, format_rna="table",
                                                                          formar_epi="table")
    # 数据归一化: 使用 normalize 函数对数据进行归一化处理，分别处理 scRNA 和 scATAC 数据。
    # 对于 scRNA 数据，使用了大小因子和对数变换；对于 scATAC 数据，仅进行基本归一化。
    x1 = normalize(x1, filter_min_counts=True,
                   size_factors=True, normalize_input=False,
                   logtrans_input=True)

    x2 = normalize(x2, filter_min_counts=True,
                   size_factors=False, normalize_input=False,
                   logtrans_input=False)

    print('===== Normalize =====')

    # 数据转化为张量
    # For scRNA
    # 获取数据矩阵: 从 x1 对象中提取实际的数据矩阵 x_scRNA。x1 是一个 AnnData 对象，x1.X 是存储表达数据的矩阵，行代表细胞，列代表基因。
    x_scRNA = x1.X
  #  np.savetxt('/data1/jyz/dataset/scMCs/file/CellMix.txt', x_scRNA)

    # 获取原始数据矩阵: 从 x1 对象中提取原始的未归一化的数据矩阵 x_scRNAraw。
    x_scRNAraw = x1.raw.X
    # 获取大小因子: 从 x1 对象的观测数据（obs）中提取大小因子 x_scRNA_size_factor。大小因子用于归一化细胞的测序深度。
    x_scRNA_size_factor = x1.obs['size_factors'].values

    # 转换为张量: 使用 torch.from_numpy 将 NumPy 数组转换为 PyTorch 张量
    x_scRNA = torch.from_numpy(x_scRNA).to(device)
    x_scRNAraw = torch.from_numpy(x_scRNAraw).to(device)
    x_scRNA_size_factor = torch.from_numpy(x_scRNA_size_factor).to(device)

    # For scATAC
    x_scATAC = x2.X

    x_scATACraw = x2.raw.X
    x_scATAC_size_factor = x2.obs['size_factors'].values

    x_scATAC = torch.from_numpy(x_scATAC).to(device)
    x_scATACraw = torch.from_numpy(x_scATACraw).to(device)
    # 获取唯一标签: 使用 np.unique 获取唯一的标签类别 classes 和标签对应的索引 label_ground_truth。
    classes, label_ground_truth = np.unique(label_ground_truth, return_inverse=True)
    classes = classes.tolist()

    N1, M1 = np.shape(x_scRNA)
    N2, M2 = np.shape(x_scATAC)

    # tensor(N1+N2, 1) \in {0, 1}
    # 生成标签张量:
    # ol_x1 和 ol_x2 分别生成大小为 (N1, 1) 的全1和全0张量，表示scRNA-seq数据的标签。
    # ol_y1 和 ol_y2 分别生成大小为 (N2, 1) 的全0和全1张量，表示scATAC-seq数据的标签。
    ol_x1 = torch.ones(N1, 1)
    ol_x2 = torch.zeros(N1, 1)
    ol_y1 = torch.zeros(N2, 1)
    ol_y2 = torch.ones(N2, 1)
    ol_x = torch.cat([ol_x1, ol_x2], dim=1).to(device)
    ol_y = torch.cat([ol_y1, ol_y2], dim=1).to(device)
    ol = torch.cat([ol_x, ol_y], dim=0).to(device)
    ol1 = ol.cpu().numpy()
    # print(ol)

    # 初始化交叉熵损失函数，用于计算分类任务的损失。
    ce_loss = nn.CrossEntropyLoss()
    if params.training_dae_scRNA:
        print("===== Pretrain a scMODF.")
        scMCs = scMODF(N1, M1, M2,
                       hidden1_1=500, hidden1_2=300, hidden1_3=128, z_emb_size1=16, dropout_rate1=0.1,
                       hidden2_1=3000, hidden2_2=2500, hidden2_3=1000, hidden2_4=128, z_emb_size2=16, dropout_rate2=0.1,
                       params=params
                       ).to(device)
     #   print(scMCs)

        # 加载预训练好的权重  pretrain_path1
        # # load the pretrained weights of ae
        DAE_ZINB_state_dict = torch.load(params.pretrain_path1)
        scMCs.DAE_ZINB.load_state_dict(DAE_ZINB_state_dict)

        DAE_Ber_state_dict = torch.load(params.pretrain_path2)
        scMCs.DAE_Ber.load_state_dict(DAE_Ber_state_dict)
        print("===== Pretrained weights are loaded successfully.")

        # 使用 Adam 优化器，并设置学习率为 0.0001。
        optimizer = Adam(scMCs.parameters(), lr=0.0005)
        train_loss_list1 = []
        ans1 = []
        ans2 = []

        for epoch in range(params.epoch1):
            total_loss = 0
            optimizer.zero_grad()  # 清零优化器的梯度缓存

            z_x, z_y, z_gx, z_gy, q_x, q_y, z_xy, z_I, y_pre, cl_loss, \
                normalized_x_zinb, dropout_rate_zinb, disper_x_zinb, scale_x_zinb, Final_x_ber, z_xy= scMCs(x_scRNA,
                                                                                                       x_scATAC,
                                                                                                       x_scRNA_size_factor)

            # ZINB loss ZINB 损失: 调用 log_zinb_positive 计算 ZINB 损失，使用均值作为最终损失值。  x_scRNA原始数据 zinb参数(scale_x_zinb disper_x_zinb dropout_rate_zinb)
            loss_zinb = torch.mean(log_zinb_positive(x_scRNA, scale_x_zinb, disper_x_zinb, dropout_rate_zinb, eps=1e-8))

            # Ber loss  二值交叉熵损失: 调用 binary_cross_entropy 计算二值交叉熵损失，使用均值作为最终损失值。
            loss_Ber = torch.mean(binary_cross_entropy(Final_x_ber, x_scATAC))

            # CE loss 交叉熵损失: 调用交叉熵损失函数 ce_loss，传入预测标签 y_pre 和真实标签 ol。标签预测器损失
            loss_ce = ce_loss(y_pre, ol)
            y_pre = y_pre.detach().cpu().numpy()

            # contrastive loss 对比损失: 直接使用前向传播返回的对比损失 cl_loss。
            loss_cl = cl_loss

            # 总损失: 计算总损失，按权重组合不同损失项。 alpha1=0.0001,alpha2=1,alpha3=0.01
            loss = loss_zinb + params.alpha1 * loss_Ber + params.alpha2 * loss_ce + params.alpha3 * loss_cl
            loss.backward()  # 反向传播: 使用 loss.backward() 进行反向传播计算梯度
            optimizer.step()  # 使用 optimizer.step() 更新模型参数。

            train_loss_list1.append(loss.item())  # 记录损失: 将当前 epoch 的损失值添加到 train_loss_list1 中，并打印损失信息。
            print("epoch {} => loss_zinb={:.4f} loss_Ber={:.4f} loss_ce={:.4f} loss_cl={:.4f} loss={:.4f}".format(epoch,
                                                                                                                  loss_zinb,
                                                                                                                  loss_Ber,
                                                                                                                  loss_ce,
                                                                                                                  loss_cl,
                                                                                                                  loss))

        # ************************************************************************************************

        # 保存和可视化结果
        print("===== save as .mat(txt) and visualization on scRNA-seq")
        z_x = z_gx.data.cpu().numpy()
        # np.savetxt('z_x.txt', z_x)
        # sio.savemat('z_x.mat', {'z_x': z_x})

        z_y = z_gy.data.cpu().numpy()
        # np.savetxt('z_y.txt', z_y)
        # sio.savemat('z_y.mat', {'z_y': z_y})

        # 将最终的综合表示 z_I 转换为 numpy 数组，并保存为文本文件 z_xxy.txt。
        z_xxy = z_I.data.cpu().numpy()


        reducer = umap.UMAP(n_neighbors=15, n_components=2, metric="euclidean")

        z_I_umap = reducer.fit_transform(z_xxy)
        # np.savetxt('z_xxy_umap.txt', z_I_umap)
        # sio.savemat('z_xxy_umap.mat', {'z_xxy_umap': z_I_umap})

        # KMeans 聚类: 使用 KMeans 进行聚类分析，指定聚类数量为 4，设置随机种子为 100。
        kmeans = KMeans(n_clusters=4, random_state=100)
        label_pred_z_I_umap = kmeans.fit_predict(z_I_umap)

        scatter = plt.scatter(z_I_umap[:, 0], z_I_umap[:, 1], c=label_ground_truth, s=10)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        plt.show()

        # 计算评价指标: 计算聚类的评价指标，包括轮廓系数 (SI)、标准化互信息 (NMI) 和调整后的兰德指数 (ARI)，并打印这些指标。
        SI_z_I = silhouette_score(z_I_umap, label_pred_z_I_umap)
        NMI_z_I = normalized_mutual_info_score(label_ground_truth, label_pred_z_I_umap, average_method='max')
        NMI_z_I = normalized_mutual_info_score(label_ground_truth, label_pred_z_I_umap, average_method='max')
        ARI_z_I = metrics.adjusted_rand_score(label_ground_truth, label_pred_z_I_umap)

        print('NMI_z_xxy = {:.4f}'.format(NMI_z_I))
        print('ARI_z_xxy = {:.4f}'.format(ARI_z_I))
        print('SI_z_xxy = {:.4f}'.format(SI_z_I))

    print("model saved to {}.".format(params.pretrain_path3 + "scMCs" + "_alpha1_" + str(params.alpha1)
                                      + "_alpha2_" + str(params.alpha2) + "_alpha3_" + str(
        params.alpha3) + "_lambda_" + str(params.lam)
                                      + "_beta_" + str(params.beta) + "_epoch_" + str(params.epoch1) + ".pkl"))
    print('===== Finished of training_scmifc =====')
