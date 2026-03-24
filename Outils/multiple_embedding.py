import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from termcolor import cprint

# 组生成器代码   通过注意力机制，将输入的嵌入特征划分成若干组，以捕捉不同的结构信息。
class QGrouping(torch.nn.Module):
    def __init__(self, args, in_emb_dim, key_dim=100):
        super(QGrouping, self).__init__()
        cprint(f'Initail a query grouping layer', 'green')
        embedding_dim = args.embedding_dim  # 192
        self.k = args.num_group  # k=3

        self.key_dim = key_dim  # 100
        self.val_dim = embedding_dim // self.k  # 每组的值维度，例如 192//3=64
        args.dim_per_group = embedding_dim // self.k  # 设置每组的维度，64
        self.att_norm = args.att_norm  # 注意力权重的归一化方式，可能是 'softmax' 或 'sigmoid'
        self.bias = args.bias  # 是否使用偏置

        if self.bias == True:
            # 定义用于生成 key, query, value 的卷积层
            self.w_k = torch.nn.Conv2d(in_emb_dim, key_dim, kernel_size=1)  # 从输入维度映射到 key_dim
            self.w_q = torch.nn.Conv2d(key_dim, self.k, 1)  # 从 key_dim 映射到组数 k
            self.w_v = torch.nn.Conv2d(in_emb_dim, self.val_dim, kernel_size=1)  # 从输入维度映射到每组的值维度
        else:
            # 如果不使用偏置，则 bias=False
            self.w_k = torch.nn.Conv2d(in_emb_dim, key_dim, kernel_size=1, bias=False)
            self.w_q = torch.nn.Conv2d(key_dim, self.k, 1, bias=False)
            self.w_v = torch.nn.Conv2d(in_emb_dim, self.val_dim, kernel_size=1, bias=False)

    def forward(self, x, batch):
        key = self.w_k(x.unsqueeze(2).unsqueeze(3))  # 在x的第二个和第三个位置新增2个维度，形状变为 (num_nodes, key_dim, 1, 1)
        val = self.w_v(x.unsqueeze(2).unsqueeze(3))  # 同上，输出形状为 (num_nodes, val_dim, 1, 1)
        val = val.squeeze()  # 移除大小为1的维度，形状变为 (num_nodes, val_dim)
        weights = self.w_q(key).reshape((-1, self.k))  # 生成查询权重，形状为 (num_nodes, k)
        norm_w = []
        embs = []

        # 遍历 batch 中的每个图
        for b in torch.unique(batch):
            if self.att_norm == 'softmax':
                this_w = F.softmax(weights[batch == b, :], dim=0)  # 对当前图的权重进行 softmax 归一化
            elif self.att_norm == 'sigmoid':
                a = torch.sigmoid(weights[batch == b, :])
                num_nodes = sum(batch == b)
                this_w = a / torch.sqrt(num_nodes.float())  # sigmoid 激活并按节点数进行归一化
            else:
                raise ValueError("Unsupported attention normalization method")

            this_val = val[batch == b, :]  # 当前图的值特征，形状为 (num_nodes_in_graph, val_dim)
            this_embs = torch.matmul(this_w.T, this_val)  # 通过矩阵乘法计算每组的聚合特征，形状为 (k, val_dim)

            embs.append(this_embs.unsqueeze(0))  # 增加一个维度，用于表示每个图的嵌入，形状为 (1, k, val_dim)
            norm_w.append(this_w)  # 保存每个节点的权重

        return torch.cat(embs, dim=0), norm_w  # 返回每个图的嵌入和对应的权重

# 将输入特征通过多个线性层映射到多个子空间（组），然后通过全局池化操作将节点特征聚合为图级别的嵌入表示。以下是对这段代码的详细解析：
class MLGrouping(torch.nn.Module):
    def __init__(self, args, in_emb_dim) -> None:
        super(MLGrouping, self).__init__()
        cprint(f'Initail a mul_linear grouping layer', 'green')
        self.pool = args.pool
        self.k = args.num_group
        self.dim_per_group = args.embedding_dim // self.k
        self.ML_ops = torch.nn.ModuleList()
        for i in range(self.k):
            self.ML_ops.append(torch.nn.Linear(in_emb_dim, self.dim_per_group))

    def forward(self, x, batch):
        out = []
        for i in range(self.k):
            fea = self.ML_ops[i](x)  # 使用第 i 个线性层对输入特征 x 进行映射
            if self.pool == 'mean':
                fea = global_mean_pool(fea, batch)  # 对特征进行全局平均池化
            else:
                fea = global_add_pool(fea, batch)  # 对特征进行全局求和池化
            out.append(fea.unsqueeze(1))  # 在第 1 维增加一个维度
        return torch.cat(out, dim=1)  # 将所有组的嵌入沿第 1 维进行拼接，返回 (num_graphs, k, dim_per_group)


