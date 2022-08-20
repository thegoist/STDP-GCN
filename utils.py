"""

训练模型时所需要的数据操作函数

"""

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score

from visdom import Visdom

import layers
from load_data_K_fold import load_all_isruc_S3, get_k_fold_data

import matplotlib.pyplot as plt

viz = Visdom(env="graph stdp k fold")


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# def f1(y_hat, y):
#
#
#     if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
#
#         y_hat = y_hat.argmax(axis=1)
#         prob = y_hat.cpu().numpy() #先把prob转到CPU上，然后再转成numpy，如果本身在CPU上训练的话就不用先转成CPU了
#         prob_all.extend(np.argmax(prob,axis=1)) #求每一行的最大值索引
#         label_all.extend(label)
#     return f1_score(y_hat.cpu().numpy(), y.cpu().numpy())

def evaluate_f1(net, data_iter, device="cuda:0"):

    prob_all = []
    label_all = []
    if isinstance(net, torch.nn.Module):
        net.eval()  # 开启评估模式，不会计算和记录累加梯度
        if not device:
            device = next(iter(net.parameters())).device
    for X, adj, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        X = X.to(torch.float32)
        adj = adj.to(device)
        y = y.to(device)

        prob = net((X, adj)) #表示模型的预测输出
        prob = prob.cpu().detach().numpy() #先把prob转到CPU上，然后再转成numpy，如果本身在CPU上训练的话就不用先转成CPU了
        prob_all.extend(np.argmax(prob, axis=1)) #求每一行的最大值索引
        label_all.extend(y.cpu().numpy())

    return f1_score(label_all, prob_all,average='macro')

def accuracy(y_hat, y):
    """
    y_hat的每一行是输出，找出每一行最大的即为预测为正确的项目
    计算预测正确的数目
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy_gpu(net, data_iter, device="cuda:0"):
    """ 使用GPU计算模型在数据集上的精度 """
    if isinstance(net, torch.nn.Module):
        net.eval()  # 开启评估模式，不会计算和记录累加梯度
        if not device:
            device = next(iter(net.parameters())).device

    metric = Accumulator(2)  # 正确预测的数量， 总预测的数量
    for X, adj, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        X = X.to(torch.float32)
        adj = adj.to(device)
        y = y.to(device)
        metric.add(accuracy(net((X, adj)), y), y.numel())

    return metric[0] / metric[1]


# https://blog.csdn.net/foneone/article/details/104445320
def k_fold_train(net, k, num_epochs, lr, batch_size, device, shuffle):
    train_acc_sum, test_acc_sum = 0, 0

    for i in range(k):
        # data = get_k_fold_data(k, i, X_train, y_train)  # 获取k折交叉验证的训练和验证数据
        psgs_concat, adjs_concat, labels_concat = load_all_isruc_S3(
            path="D:\\data\\ISRUC_S3\\ISRUC_S3_features_adjs_re.npz", shuffle=shuffle)
        train_iter, test_iter = get_k_fold_data(k, i, batch_size, psgs_concat, adjs_concat, labels_concat)

        train_acc, test_acc = train(net, train_iter, test_iter, num_epochs, lr, device=device, k=k, k_i=i)

        print('*' * 25, '第', i + 1, '折', '*' * 25)
        print('train_loss:%.6f' % train_acc, 'test_acc:%.4f\n' % test_acc)

        train_acc_sum += train_acc
        test_acc_sum += test_acc

    print('#' * 10, '最终k折交叉验证结果', '#' * 10)
    ####体现步骤四#####
    print('train_acc_sum:%.4f\n' % (train_acc_sum / k), 'test_acc_sum:%.4f' % (test_acc_sum / k))


def train(net, train_iter, test_iter, num_epochs, lr, device, k=0, k_i=0):
    """
    重写为训练GCN的训练模式
    :param net:构建完成的网络
    :param train_iter: 训练数据
    :param test_iter: 测试数据
    :param num_epochs:
    :param lr: 学习率
    :param device: gpu
    :return:
    """

    viz.line([[0.5, 0.5]], [0.], win=f'acc {k} fold {k_i}', opts=dict(
        title=f'acc {k} fold {k_i}', legend=['train_acc', 'test_acc']))

    best_train_acc = 0
    best_test_acc = 0

    # 初始化网络权重
    # 还要初始化一些图神经的权重
    # 图神经
    def init_weights(m):
        # print(1, type(m))
        if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == layers.GraphConvolution:
            # print(2, type(m))
            nn.init.xavier_normal_(m.weight)

    net.apply(init_weights)

    # num_batches = len(train_iter)
    print('training on', device)


    net.to(device)

    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    # loss = gl_loss()

    # train
    cnt = 0
    for epoch in range(num_epochs):
        net.train()

        metric = Accumulator(3)  # 这里是为了简洁加了累加器
        train_acc = []
        for i, (X, adj, y) in enumerate(train_iter):
            # X, y = X.to(device), y.to(device)
            # plt.imshow(X[0][0])
            # plt.show()
            X, adj, y = X.float().to(device), adj.to(device), y.to(device)
            optimizer.zero_grad()
            X = X.float().to(device)

            y_hat = net((X, adj))

            l = loss(y_hat, y.long())

            l.backward()
            optimizer.step()

            with torch.no_grad():
                # 注意这里， 第一个元素的含义是什么，batch内数据的数量，将Loss乘数量
                # metric[0]是累加loss乘batch数量， metric[1]是累加正确的y的数量， metric[2]是batch数累加
                metric.add(l * X.shape[0], accuracy(y_hat, y.long()), X.shape[0])

            train_l = metric[0] / metric[2]  # 求平均loss
            train_acc = metric[1] / metric[2]
            cnt += 1
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            # viz.line([[train_acc, train_l]], [cnt], win='loss', update='append')

            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     print(f'epoch: {epoch} train loss: {train_l}, train_acc {train_acc}')
            # print(f'epoch: {epoch} train loss: {train_l}, train_acc {0}')

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        test_f1 = evaluate_f1(net, test_iter)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        viz.line([[train_acc, test_acc]], [epoch], win=f'acc {k} fold {k_i}', update='append')
        print(f'{k}折 第{k_i}折 epoch: {epoch} loss {train_l:.3f}, train acc {train_acc:.3f}',
              f'test acc {test_acc:.3f}',
              f'test f1 {test_f1:.3f}',
              f'best acc {best_test_acc:.3f}',
              f'on {str(device)}')
        # print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')

    return best_train_acc, best_test_acc
