import numpy as np
from sklearn import cluster
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

nmi = normalized_mutual_info_score
ami = adjusted_mutual_info_score
ari = adjusted_rand_score

# 定义了一些用于聚类性能评估和后处理的函数，包括邓恩指数计算、杰卡尔德系数计算、聚类精度计算、错误率计算、稀疏化和后处理函数。


# 邓恩指数
# 计算聚类结果的邓恩指数，衡量聚类的紧密性和分离性。
# data：数据点。
# a：聚类标签。
def DI_calcu(data, a):
    dmin = float("inf")
    maxdiam = 0
    b = len(data)
    for i in range(b-1):
        for j in range(i+1, b):
            if ((data[i]-data[j])**2).sum()<dmin and a[i] != a[j]:
                dmin=((data[i]-data[j])**2).sum()
            if ((data[i]-data[j])**2).sum()>maxdiam and a[i] == a[j]:
                maxdiam = ((data[i]-data[j])**2).sum()
    c = (dmin/maxdiam)**0.5
    return c


# 杰卡尔德系数
# 计算聚类结果的杰卡尔德系数，衡量聚类结果与真实标签的相似度。
# gnd2：真实标签。
# pred：预测标签。
def JC_calcu(gnd2, pred):
    lengnd = len(gnd2)
    aj, bj, cj, dj = 0.0, 0.0, 0.0, 0.0
    for i in range(lengnd - 1):
        for j in range(i + 1, lengnd):
            if gnd2[i] == gnd2[j] and pred[i] == pred[j]:
                aj = aj + 1
            elif gnd2[i] == gnd2[j] and pred[i] != pred[j]:
                bj = bj + 1
            elif gnd2[i] != gnd2[j] and pred[i] == pred[j]:
                cj = cj + 1
            elif gnd2[i] != gnd2[j] and pred[i] != pred[j]:
                dj = dj + 1
    jc = aj/(aj+bj+cj)
    return jc

# 计算聚类精度，通过线性分配问题解决标签匹配问题。
# y_true：真实标签。
# y_pred：预测标签。
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size

# 计算聚类错误率。
# gt_s：真实标签。
# s：预测标签。
def err_rate(gt_s, s):
    return 1.0 - acc(gt_s, s)

# 对系数矩阵 C 进行稀疏化。
# C：系数矩阵。
# alpha：稀疏化参数。
def thrC(C, alpha):
    if alpha < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > alpha * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp

# 对系数矩阵 C 进行后处理，进行光谱聚类。
# C：系数矩阵。
# K：聚类数量。
# d：每个子空间的维度。
# ro：矩阵的幂指数。
def post_proC(C, K, d, ro):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    n = C.shape[0]
    C = 0.5 * (C + C.T)
    # C = C - np.diag(np.diag(C)) + np.eye(n, n)  # good for coil20, bad for orl
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(n))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** ro)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L)
    return grp, L

# 执行光谱聚类。
# C：系数矩阵。
# K：聚类数量。
# d：每个子空间的维度。
# alpha：稀疏化参数。
# ro：矩阵的幂指数。
def spectral_clustering(C, K, d, alpha, ro):
    C = thrC(C, alpha)
    y, _ = post_proC(C, K, d, ro)
    return y
