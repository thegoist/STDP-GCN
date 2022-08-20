"""
用于定义GCN模型的训练过程
包括数据加载，参数初始化以及训练方法

"""

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from layers import BatchNorm
from load_data import load_isruc_S3

from models import GCN, GraphConvStatic, GraphConvSTDP, GraphConvStdpWithAdj, StdpGCN, StdpGCN_test, StdpGCN_context

from utils import train, k_fold_train

# 参数设置
# 这部分是准确率最终能到87的
# parser = argparse.ArgumentParser()
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training.')  # action='store_true'表示只要运行配置这个选项就设置为True,
# parser.add_argument('--fastmode', action='store_true', default=False,
#                     help='validate during training pass')
# parser.add_argument('--seed', type=int, default=42, help='random seed')
# parser.add_argument('--epochs', type=int, default=500000, help="number of epochs to train")
# parser.add_argument('--batch_size', type=int, default=256, help="number of batch")
# parser.add_argument('--lr', type=float, default=0.001, help="initial learning rate")
# parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay l2 loss on parameters")
# parser.add_argument('--hidden', type=int, default=16, help='number of hidden units')
# parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate 1-keep probability')

# 调参实验
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training.')  # action='store_true'表示只要运行配置这个选项就设置为True,
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='validate during training pass')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--epochs', type=int, default=1000, help="number of epochs to train")
parser.add_argument('--batch_size', type=int, default=256, help="number of batch")
parser.add_argument('--lr', type=float, default=0.0002, help="initial learning rate")
parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay l2 loss on parameters")
parser.add_argument('--shuffle', type=bool, default=True, help='dropout rate 1-keep probability')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate 1-keep probability')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 设置随机数种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# train_iter, test_iter = load_isruc_S3(batch_size=args.batch_size, path="D:\\data\\ISRUC_S3\\ISRUC_S3.npz")
# train_iter, test_iter = load_isruc_S3(batch_size=args.batch_size,
#                                       path="D:\\data\\ISRUC_S3\\ISRUC_S3_features_origin.npz")

net = nn.Sequential(
    # StdpGCN_test(nfeat=9, nid=256, nclass=9, dropout=0.5),
    StdpGCN_context(nfeat=9, nid=256, nclass=9, dropout=0.5),
    # StdpGCN(nfeat=9, nhid=256, nclass=9, dropout=0.5, eeg_size=3000),
    # GraphConvStdpWithAdj(nfeat=9, nhid=256, nclass=32, dropout=0.5, eeg_size=3000),
    # GraphConvSTDP(nfeat=18, nhid=256, nclass=32, dropout=0.5, eeg_size=1000),
    # GraphConvStatic(nfeat=384, nhid=256, nclass=128, dropout=0.5, adj=adj),
    # GraphConvStatic(nfeat=128, nhid=64, nclass=32, dropout=0.5, adj=adj),

    nn.Flatten(),
    nn.Linear(450, 1024), nn.ReLU(), nn.Dropout(p=0.5),
    # # nn.Linear(512, 512), nn.ReLU(),nn.Dropout(p=0.5),
    nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(1024, 5), nn.Softmax()
    # nn.Linear(90, 5), nn.Softmax()
)
X = torch.randn([1, 5, 10, 9])
adjs = torch.randn([1, 5, 10, 10])

# print(net((X, adjs)).shape)
for layer in net:
    if isinstance(layer, StdpGCN_context):
        X = layer((X, adjs))
    else:
        X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
# wd = net[2].weight.norm().item()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.001)
net = net.to("cuda:0")
# input = torch.randn([50, 10, 18])
# print(net(input).shape)

# 不k折交叉验证训练程序
# train_iter, test_iter = load_isruc_S3(batch_size=args.batch_size,
#                                       path="D:\\data\\ISRUC_S3\\ISRUC_S3_features_adjs_re.npz", shuffle=True)
# train(net, train_iter, test_iter, args.epochs, args.lr, device="cuda:0")

k_fold_train(net, 10, num_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device="cuda:0",
             shuffle=args.shuffle)
