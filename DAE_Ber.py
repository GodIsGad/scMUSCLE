_author_ = 'lrren'
# coding: utf-8

_author_ = 'lrren'
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

# 这段代码定义了一个名为 DAE_Ber 的深度自动编码器（Deep AutoEncoder）。
# 该模型由编码器和解码器两个部分组成，用于学习数据的低维表示并重建数据。下面是详细的代码解读：

# 参数:
# in_dim: 输入数据的维度。
# hidden1, hidden2, hidden3, hidden4: 隐藏层的神经元数量。
# z_emb_size: 潜在空间的维度。
# dropout_rate: Dropout 层的丢弃率，用于防止过拟合。
# 定义编码器 fc_encoder:
# 使用 nn.Sequential 顺序构建一系列全连接层（nn.Linear）、激活函数（nn.ReLU）和 Dropout 层（nn.Dropout）。
# 定义解码器 fc_decoder:
# 使用类似编码器的方法构建全连接层、激活函数和 Dropout 层。
# 定义最终的线性层 decoder_scale 和激活函数 sig:
# decoder_scale 用于将解码器输出映射回原始输入维度。
# sig 使用 Sigmoid 激活函数。
class DAE_Ber(nn.Module):

    def __init__(self, in_dim, hidden1, hidden2, hidden3, hidden4, z_emb_size, dropout_rate):
        super(DAE_Ber, self).__init__()

        # self.params = args
        self.in_dim = in_dim

        self.fc_encoder = nn.Sequential(
            nn.Linear(self.in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden3, hidden4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden4, z_emb_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # Distribution reconstruction
        self.fc_decoder = nn.Sequential(
            nn.Linear(z_emb_size, hidden4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden4, hidden3),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden3, hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        self.decoder_scale = nn.Linear(hidden1, self.in_dim)
        self.sig = nn.Sigmoid()



# 前向传播 forward
    # 输入参数:
        # x: 输入数据。
        # scale_factor: 缩放因子，默认为1.0。
    # 编码器:
        # 将输入数据 x 传入编码器 fc_encoder，得到编码后的表示 emb。
    # 解码器:
        # 将编码后的表示 emb 传入解码器 fc_decoder，得到解码后的表示 latent。
    # 重建数据:
        # 将解码后的表示 latent 传入 decoder_scale，得到重建的数据 recon_x。
        # 对重建的数据 recon_x 进行 Sigmoid 激活，得到最终的重建数据 Final_x。
    # 返回:
        # 返回编码后的表示 emb 和最终的重建数据 Final_x。
    def forward(self, x, scale_factor = 1.0 ):

        emb = self.fc_encoder(x)
        # emb(1047,16)
        # expression matrix decoder
        latent = self.fc_decoder(emb)
        # (1047,3000)
        recon_x = self.decoder_scale(latent)
        # (1047,7136)
        Final_x = self.sig(recon_x)
        # (1047,7136)

        return emb, Final_x
