"""

定义图卷积层，图卷积层包括两个操作：邻居聚合与特征变换。
-邻居聚合：用于聚合邻居结点的特征。
-特征变换：传统NN的操作，即特征乘参数矩阵。

基于pytorch实现，需要导入torch中的parameter和module模块。


"""
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.nn.modules.module import Module


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断训练模式还是预测模式
    if not torch.is_grad_enabled():
        """在预测模式下，不可得到当前miniBatch的均值
        这里的均值和方差是从全局获得，
        一般是指数加权平均计算得到，这种平均方法可以不记住所有的值求均值，还能得到和均值相似的值
        """
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        """
        assert 判断当前层必须是全连接或者2d卷积?其余情况此函数不处理
        X的形状是2，说明是展平进行全连接
        X的形状是4，说明是多通道的卷积
        然后根据是全连接还是卷积进行相应的处理
        """
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        """
        注意这里更新移动平均的均值和方差
        使用了指数加权平均
        """
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var

    Y = gamma * X_hat + beta

    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    # `num_features`：完全连接层的输出数量或卷积层的输出通道数。
    # `num_dims`：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        """
        这里由于gama beta需要求梯度，参与反向传播，因此放入Parameter里面，是模型的参数
        而moving_mean moving_var这些
        """
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果 `X` 不在内存上，将 `moving_mean` 和 `moving_var`
        # 复制到 `X` 所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta,
            self.moving_mean, self.moving_var,
            eps=1e-5, momentum=0.9
        )

        return Y


class GraphConvolution(Module):
    """
    图卷积层的作用是接收旧特征并产生新特征
    因此初始化的时候需要确定两个参数：输入特征的维度 与 输出特征的维度
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # parameter 作用是将tensor设置为梯度求解 就可以自动求导啦，并将其绑定到模型的参数中。
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 参数的初始化

    def reset_parameters(self):
        # nn.init.xavier_normal_(self.weight)
        # if self.bias is not None:
        #     nn.init.xavier_normal_(self.bias)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # 这两部主要是扩充成统一维度
        # print(f'1 input shape {input.shape}, adj shape{adj.shape} self weight {self.weight.shape}')

        # self.weight.data = self.weight.data.unsqueeze(0)
        # self.weight.data = self.weight.data.repeat(input.shape[0], 1, 1)
        # print(f'1 input shape {input.shape}, adj shape{adj.shape} self weight {self.weight.shape}')

        # support = torch.bmm(input, self.weight)  # 特征变换
        # output = torch.bmm(adj, support)  # 邻居聚合
        support = input @ self.weight  # 特征变换
        # print(f'2 support shape {support}, adj shape{adj.shape} self weight {self.weight.shape}')
        # support = support.to(torch.float32)
        # adj = adj.to(torch.float32)
        output = adj @ support  # 邻居聚合
        # print(f'2 support shape {support.shape}, adj shape{adj.shape} output {output.shape}')
        # self.weight.data = self.weight.data[0]
        # print(f'3 input shape {input.shape}, adj shape{adj.shape} self weight {self.weight.shape}')

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # 打印信息
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


if __name__ == '__main__':
    input = torch.arange(0, 4).resize(2, 2).float()
    input = input.repeat(2, 1, 1)
    adj = torch.ones(2, 2)
    gcn = GraphConvolution(in_features=2, out_features=2)
    print(gcn(input, adj))
