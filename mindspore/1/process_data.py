import scanpy as sc
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

#这些函数主要用于处理和标准化单细胞 RNA-seq 和表观遗传学数据。


# 该函数用于对数据进行主成分分析（PCA）。
# 输入 X 是原始数据矩阵，dim 是目标维度。
# 使用 PCA 将数据降维到 dim 维度，返回降维后的数据 X_10。
def dopca(X, dim):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10



# 该函数用于读取单细胞多组学数据。
# File1、File2 分别为 scRNA-seq 和 scEpigenomics 数据文件路径。
# File3、File4 为细胞分组信息文件路径。
# 根据不同的 state 读取分组信息。
# 如果 test_size_prop 大于 0，则划分训练集和测试集。
# 返回值包括处理后的数据、训练集和测试集索引、以及标签信息
def read_dataset(File1=None, File2=None, File3=None, File4=None, transpose=True, test_size_prop=None, state=0,
                 format_rna=None, formar_epi=None):
    # read single-cell multi-omics data together

    ### raw reads count of scRNA-seq data  scRNA-seq数据的原始读取计数

    adata = adata1 = None

    #  读取 scRNA-seq 数据
    if File1 is not None:
        if format_rna == "table":
            adata = sc.read(File1)
        else:  # 10X format
            adata = sc.read_mtx(File1)

        if transpose:
            adata = adata.transpose()

    #   读取 scEpigenomics 数据
    ##$ the binarization data for scEpigenomics file  scEpigenomics文件二值化数据
    if File2 is not None :
        if formar_epi == "table":
            adata1 = sc.read(File2)
        else:  # 10X format
            adata1 = sc.read_mtx(File2)

        if transpose:
            adata1 = adata1.transpose()

    # 读取细胞分组信息
    ### File3 and File4 for cell group information of scRNA-seq and scEpigenomics data
    label_ground_truth = []
    label_ground_truth1 = []

    if state == 0:
        if File3 is not None:
            Data2 = pd.read_csv(File3, header=0, index_col=0)
            label_ground_truth = Data2['Group'].values

        else:
            label_ground_truth = np.ones(len(adata.obs_names))

        if File4 is not None:
            Data2 = pd.read_csv(File4, header=0, index_col=0)
            label_ground_truth1 = Data2['Group'].values

        else:
            label_ground_truth1 = np.ones(len(adata.obs_names))

    elif state == 1:
        if File3 is not None:
            Data2 = pd.read_table(File3, header=0, index_col=0)
            label_ground_truth = Data2['cell_line'].values
        else:
            label_ground_truth = np.ones(len(adata.obs_names))

        if File4 is not None:
            Data2 = pd.read_table(File4, header=0, index_col=0)
            label_ground_truth1 = Data2['cell_line'].values
        else:
            label_ground_truth1 = np.ones(len(adata.obs_names))

    elif state == 3:
        if File3 is not None:
            Data2 = pd.read_table(File3, header=0, index_col=0)
            label_ground_truth = Data2['Group'].values
        else:
            label_ground_truth = np.ones(len(adata.obs_names))

        if File4 is not None:
            Data2 = pd.read_table(File4, header=0, index_col=0)
            label_ground_truth1 = Data2['Group'].values
        else:
            label_ground_truth1 = np.ones(len(adata.obs_names))

    else:
        if File3 is not None:
            Data2 = pd.read_table(File3, header=0, index_col=0)
            label_ground_truth = Data2['Cluster'].values
        else:
            label_ground_truth = np.ones(len(adata.obs_names))

        if File4 is not None:
            Data2 = pd.read_table(File4, header=0, index_col=0)
            label_ground_truth1 = Data2['Cluster'].values
        else:
            label_ground_truth1 = np.ones(len(adata.obs_names))

    # 划分训练集和测试集
    # split datasets into training and testing sets
    if test_size_prop > 0:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs),
                                               test_size=test_size_prop,
                                               random_state=200)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['split'] = spl.values

        if File2 is not None:
            adata1.obs['split'] = spl.values
    else:
        train_idx, test_idx = list(range(adata.n_obs)), list(range(adata.n_obs))
        spl = pd.Series(['train'] * adata.n_obs)
        adata.obs['split'] = spl.values

        if File2 is not None:
            adata1.obs['split'] = spl.values

    adata.obs['split'] = adata.obs['split'].astype('category')
    adata.obs['Group'] = label_ground_truth
    adata.obs['Group'] = adata.obs['Group'].astype('category')

    if File2 is not None:
        adata1.obs['split'] = adata1.obs['split'].astype('category')
        adata1.obs['Group'] = label_ground_truth
        adata1.obs['Group'] = adata1.obs['Group'].astype('category')

    # print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    ### here, adata with cells * features
    return adata, adata1, train_idx, test_idx, label_ground_truth, label_ground_truth1

# 这个函数 normalize 主要用于对单细胞数据进行预处理和标准化。函数接受一个 AnnData 对象 adata，并根据不同的参数进行不同的处理步骤。
def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True):
    if filter_min_counts:
        # 过滤掉计数低于 1 的基因和细胞。
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    # 决定是否备份原始数据到 adata.raw
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if logtrans_input:
        sc.pp.log1p(adata)

    if size_factors:
        adata.obs['size_factors'] = np.log(np.sum(adata.X, axis=1))
    else:
        adata.obs['size_factors'] = 1.0

    # 如果 normalize_input 为 True，则对数据进行标准化。
    if normalize_input:
        sc.pp.scale(adata)

    return adata

# 这个函数 normalize2 用于对单细胞 RNA-seq 数据进行预处理和标准化。
# 该函数提供了更丰富的选项来处理数据，包括从文件加载数据、过滤基因和细胞、计算 size factors、对数转换、选择高变异基因和标准化。
def normalize2(adata, copy=True, highly_genes=None, filter_min_counts=True, size_factors=True, normalize_input=True,
               logtrans_input=True):
    # adata: 一个 AnnData 对象或文件路径（字符串），包含单细胞 RNA-seq 数据。
    # copy (默认值: True): 是否复制 AnnData 对象。
    # highly_genes (默认值: None): 用于选择高变异基因的数量。
    # filter_min_counts (默认值: True): 是否过滤掉计数低于阈值的基因和细胞。
    # size_factors (默认值: True): 是否计算和添加 size factors，用于标准化数据。
    # normalize_input (默认值: True): 是否标准化输入数据。
    # logtrans_input (默认值: True): 是否对数据进行对数转换。

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    # 如果 adata 是 AnnData 对象且 copy 为 True，则复制该对象。
    # 如果 adata 是字符串，则将其视为文件路径并加载数据。
    # 如果既不是 AnnData 对象也不是字符串，则抛出未实现错误。

    # 检查数据集中是否包含未标准化的计数数据
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6:  # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    # 如果 filter_min_counts 为 True，则过滤掉计数低于 1 的基因和细胞。
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes,
                                    subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata
