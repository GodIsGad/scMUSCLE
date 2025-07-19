import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--pretrain_path1', type=str,default='/data/jyz/scMCs/pkl_baseLine_0.89_0.92/dae_scRNA.pkl')
    parser.add_argument('--pretrain_path2', type=str,default='/data/jyz/scMCs/pkl_baseLine_0.89_0.92/dae_scATAC.pkl')
    # 保存加载的数据
    parser.add_argument('--pretrain_path3', type=str,default='/data/jyz/scMCs/file/')
    parser.add_argument('--epoch1', type=int, default=500, help='Number of epochs to training_model.')
    #  parser.add_argument('--training_dae_scRNA', type=bool, default=True, help='Training dae.')
    parser.add_argument('--training_dae_scRNA', action='store_true', help='Whether to train the dae model for scRNA.')

    # parameters for multi-omics data fusion
    # 多组学数据融合参数: 添加参数 --lam 和 --beta 分别指定组学融合的权重。
    parser.add_argument('--lam', type=int, default=0.1, help='omics fusion for Z_{gY}')
    parser.add_argument('--beta', type=int, default=1, help='omics fusion for Z_{XY}')
    # parameters for model optimization
    # 模型优化参数: 添加参数 --alpha1、--alpha2 和 --alpha3 分别指定不同损失项的权重。
    parser.add_argument('--alpha1', type=int, default=0.0001, help='weight of loss_ber')
    parser.add_argument('--alpha2', type=int, default=1, help='weight of loss_dis')
    parser.add_argument('--alpha3', type=int, default=0.01, help='weight of loss_cl')
    parser.add_argument('--local', dest='local', action='store_const',const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const',const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const',const=True, default=False)

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
    parser.add_argument('--lam_div', default=0.4, type=float)
    parser.add_argument('--num_group', default=3,type=int, help='reduction ratio of key')

    parser.add_argument('--top_k', default=11, type=int)
    parser.add_argument('--device', default=0, type=int)


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

    return parser.parse_args()


