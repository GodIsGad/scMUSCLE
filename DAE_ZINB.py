_author_ = 'lrren'
# coding: utf-8

_author_ = 'lrren'
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
# 这个模型主要用于从输入数据中学习到低维的表示（潜在表示），并能够从低维表示重建出原始数据，同时考虑到零膨胀负二项分布的特性。
# Dropout 层有助于防止过拟合，ReLU 激活函数用于引入非线性变换，softmax 激活函数用于对输出进行标准化处理。

# 这段代码定义了一个名为 DAE_ZINB 的深度自动编码器（Deep AutoEncoder），用于处理零膨胀负二项分布（ZINB）模型。
# 该模型由编码器和解码器两个部分组成，用于学习数据的低维表示并重建数据，同时考虑到零膨胀特性。以下是详细的代码解读：

# 参数:
    # in_dim: 输入数据的维度。
    # hidden1, hidden2, hidden3: 隐藏层的神经元数量。
    # z_emb_size: 潜在空间的维度。
    # dropout_rate: Dropout 层的丢弃率，用于防止过拟合。
    # 定义编码器 fc_encoder:
    # 使用 nn.Sequential 顺序构建一系列全连接层（nn.Linear）、激活函数（nn.ReLU）和 Dropout 层（nn.Dropout）。
    # 定义解码器 fc_decoder:
    # 使用类似编码器的方法构建全连接层、激活函数和 Dropout 层。
    # 定义解码器的其他层:
    # decoder_scale 用于将解码器输出映射回原始输入维度。
    # decoder_r 用于计算分布的参数。
    # dropout 用于计算 dropout 率。
class DAE_ZINB(nn.Module):

    def __init__(self, in_dim, hidden1, hidden2, hidden3, z_emb_size, dropout_rate):
        super(DAE_ZINB, self).__init__()

        # self.params = args
        self.in_dim = in_dim

        self.fc_encoder = nn.Sequential(
            nn.Linear(self.in_dim, hidden1),
            # ReLU提供非线性变换，使网络能够处理复杂的非线性关系。
            nn.ReLU(),
            # Dropout提供正则化，防止网络过拟合，提高泛化能力。
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden3, z_emb_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # Distribution reconstruction  分布重建 基于ZINB的解码器的3个估计参数基于ZI通过三个不同的全连接层如下
        self.fc_decoder = nn.Sequential(
            nn.Linear(z_emb_size, hidden3),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden3, hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
# 3个估计参数由三个不同全连接层实现
        self.decoder_scale = nn.Linear(hidden1, self.in_dim)
        self.decoder_r = nn.Linear(hidden1, self.in_dim)
        self.dropout = nn.Linear(hidden1, self.in_dim)

    # 前向传播 forward
        # 输入参数:
            # x: 输入数据。
            # scale_factor: 缩放因子，默认为1.0。
        # 编码器:
            # 将输入数据 x 传入编码器 fc_encoder，得到编码后的表示 emb。
        # 解码器:
            # 将编码后的表示 emb 传入解码器 fc_decoder，得到解码后的表示 latent。
        # 重建数据:
            # 将解码后的表示 latent 传入 decoder_scale，通过 softmax 激活函数标准化，得到 normalized_x。
            # 调整 scale_factor 的尺寸，并扩展其维度以匹配 normalized_x。
            # 使用 scale_factor 对 normalized_x 进行缩放，得到 scale_x。
            # 计算分布参数 disper_x（用 decoder_r 和 exp）。
            # 计算 dropout 率 dropout_rate。
        # 返回:
          # 返回编码后的表示 emb，标准化的重建数据 normalized_x，dropout 率 dropout_rate，分布参数 disper_x，以及缩放后的重建数据 scale_x。

    def forward(self, x, scale_factor = 1.0 ):
        # 编码器：将输入数据 x 传入编码器，得到编码后的表示 emb
        emb = self.fc_encoder(x)

        # expression matrix decoder
        # 解码器：将编码后的表示 emb 传入解码器，得到解码后的表示 latent
        latent = self.fc_decoder(emb)
        # 生成标准化的重建数据
        normalized_x = F.softmax(self.decoder_scale(latent), dim=1)

        # 调整 scale_factor 的尺寸，并扩展其维度以匹配 normalized_x
        batch_size = normalized_x.size(0)
        scale_factor.resize_(batch_size, 1)

        # 使用 scale_factor 对 normalized_x 进行缩放，得到 scale_x
        scale_factor.repeat(1, normalized_x.size(1))

        scale_x = torch.exp(scale_factor) * normalized_x  # recon_x
        # scale_x = normalized_x  # recon_x
        disper_x = torch.exp(self.decoder_r(latent))  ### theta
        dropout_rate = self.dropout(latent)

        return emb, normalized_x, dropout_rate, disper_x, scale_x