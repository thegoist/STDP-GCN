import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from STDP_graph import GraphConstructSTDP
from layers import GraphConvolution


class StdpGCN_context(nn.Module):
    """
    增加了上下文，在时间方向使用卷积
    其实有点类似TCN？
    效果目前 没有加跳步连接 18814次迭代，训练ACC 86, 测试ACC 80.5
           加跳步连接    7460迭代，   训练ACC 89, 测试ACC 86
    """

    def __init__(self, nfeat, nid, nclass, dropout):
        """
                :param nfeat: 输入的特征数量
                :param nhid: 隐藏层的特征数量
                :param nclass: 分类数
                :param eeg_size: 一个用于分界eeg信号和特征信息的数
                :param dropout:
                """
        super(StdpGCN_context, self).__init__()
        self.dropout = dropout
        self.gcn = GCN(nfeat, nid, nclass, dropout)
        self.bn = nn.BatchNorm2d(9)

        self.time_conv = nn.Conv2d(in_channels=nfeat, out_channels=nclass, kernel_size=(1, 3), stride=(1, 1),
                                   padding=(0, 1))

    def forward(self, input):
        """

        :param input: x=[batch,context, channel, encoded eeg + features],
                      adjs=[batch, channel, channel]
        :return:
        """
        features, adjs = input  # DE 特征

        # 测试 随机rand adj效果
        # adj = torch.randn(adjs.shape) > 0
        ones_adj = torch.ones(adjs.shape).float().to(features.device)
        # random_adj = adj.float().to(features.device)
        # print(adj[0])

        timesteps = features.shape[1]
        x = torch.zeros_like(features)
        for t in range(1, timesteps):
            # x[:, t, :, :] = self.gcn(features[:, t, :, :], adjs[:, t, :, :])
            # x[:, t, :, :] = self.gcn(features[:, t, :, :], random_adj[:, t, :, :])  # 测试随机矩阵
            x[:, t, :, :] = self.gcn(features[:, t, :, :], ones_adj[:, t, :, :])  # 测试全1矩阵

        # print(f'context GCN shape{x.shape}')
        # output shape [batch, 1, channel, features']
        x_time = rearrange(x, 'b c h w -> b w h c')
        # print(f'rearrange shape{x_time.shape}')

        x_time_conv = self.time_conv(x_time)
        x_time_conv = self.bn(x_time_conv)
        # print(f'x_time_conv shape{x_time_conv.shape}')
        x_time_conv = rearrange(x_time_conv, 'b w h c -> b c h w')
        # print(f'x_time_conv rearrange shape{x_time_conv.shape}')

        # x_residual = self.short_cut(x)
        # print(f'x_residual shape{x_residual.shape}')
        # 跳步连接
        output = features + x_time_conv

        # return F.log_softmax(x_time_conv, dim=1)
        return output


class StdpGCN_test(nn.Module):
    """
     这套网络实现了84的测试准确率
     没有使用上下文信息
    """

    def __init__(self, nfeat, nid, nclass, dropout):
        """
                :param nfeat: 输入的特征数量
                :param nhid: 隐藏层的特征数量
                :param nclass: 分类数
                :param eeg_size: 一个用于分界eeg信号和特征信息的数
                :param dropout:
                """
        super(StdpGCN_test, self).__init__()
        self.dropout = dropout
        self.gcn = GCN(nfeat, nid, nclass, dropout)

    def forward(self, input):
        """

        :param input: x=[batch,context, channel, encoded eeg + features], adjs=[batch, channel, channel]
        :return:
        """
        features, adjs = input  # 选择 de 特征

        # output shape [batch,  channel, features]
        x = self.gcn(features, adjs)

        return x


class StdpGCN(nn.Module):
    """
    针对有上下时间的数据 进行图卷积
    图由STDP计算得出
    数据目前为DE分离出的EEG特征

    网络参数设定：
    图卷积层数量
    图卷积层的相关参数
    分类数

    时空两个维度进行的处理
    时空注意力
    时空卷积

    """

    def __init__(self, nfeat, nhid, nclass, dropout, eeg_size):
        """
                :param nfeat: 输入的特征数量
                :param nhid: 隐藏层的特征数量
                :param nclass: 分类数
                :param eeg_size: 一个用于分界eeg信号和特征信息的数
                :param dropout:
                """
        super(StdpGCN, self).__init__()
        self.eeg_size = eeg_size
        self.gcn = GCN(nfeat, nhid, nclass, dropout)
        self.time_conv = nn.Conv2d(in_channels=nfeat, out_channels=nclass, kernel_size=(1, 3), stride=(1, 1),
                                   padding=(1, 1))
        self.short_cut = nn.Conv2d(in_channels=nfeat, out_channels=5, kernel_size=1, stride=1, padding=(0, 1))

    def forward(self, input):
        """

        :param input: x=[batch,context, channel, encoded eeg + features], adjs=[batch, channel, channel]
        :return:
        """
        features, adjs = input  # 选择 de 特征
        # 只选一个epoch
        # features = x[:, 2, :, :]
        # adjs = adjs[:, 2, :, :]

        # output shape [batch, context, channel, features]
        x = self.gcn(features, adjs)

        # print(f'context GCN shape{x.shape}')
        # output shape [batch, 1, channel, features']
        # x_time = rearrange(x, 'b c h w -> b w h c')
        # print(f'rearrange shape{x_time.shape}')

        # x_time_conv = self.time_conv(x_time)
        # x_time_conv = rearrange(x_time_conv, 'b w h c -> b c h w')
        # print(f'x_time_conv shape{x_time_conv.shape}')

        # x_residual = self.short_cut(x)
        # print(f'x_residual shape{x_residual.shape}')

        # output = x_residual + x_time_conv

        # return F.log_softmax(x_time_conv, dim=1)
        return x


class GraphConvStdpWithAdj(nn.Module):
    """
    本类处理的是邻接图已经预先算好
    输入预先算好的STDP的图
    """

    def __init__(self, nfeat, nhid, nclass, dropout, eeg_size):
        """
        :param nfeat: 输入的特征数量
        :param nhid: 隐藏层的特征数量
        :param nclass: 分类数
        :param eeg_size: 一个用于分界eeg信号和特征信息的数
        :param dropout:
        """
        super(GraphConvStdpWithAdj, self).__init__()
        self.eeg_size = eeg_size

        # # 外部定义好图的样式
        # self.adj_by_STDP = []
        #
        # self.adjs_constructor = GraphConstructSTDP(device="cuda:0")
        # 两层卷积层
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = dropout

    def forward(self, input):
        """
        :param input: x=[batch, epoch, channel, encoded eeg +features], adjs=[batch, channel, channel]
        :return:
        """
        x, adjs = input
        # print(adjs)
        with torch.no_grad():
            batch_size = x.shape[0]
            eeg = x[:, :, :self.eeg_size]
            features = x[:, :, :, self.eeg_size + 9:]
            # print(f'eeg shape : {features.shape}')
            # print(f'x device : {x.device}')

            # # todo 这里不再自动计算图
            self.adj_by_STDP = adjs  # 已经预先计算好图
            # self.adj_by_STDP = self.adjs_constructor.get_batch_graph(input=eeg).to(torch.float32)
            # self.adj_by_STDP = adjs_constructor.get_batch_graph_multiprocessing(input=eeg).to(torch.float32)

        x = F.relu(self.gc1(features, self.adj_by_STDP))  # 第一层卷积
        x = F.dropout(x, self.dropout, training=self.training)  # 只在训练的时候启动
        x = self.gc2(x, self.adj_by_STDP)
        # print(x.shape)
        # print(x.shape)
        return F.log_softmax(x, dim=1)


class GraphConvSTDP(nn.Module):
    # TODO 数据的预处理，针对[batch, epoch, channel, eeg+feature]类型的数据进行处理 (现在还没完成）

    def __init__(self, nfeat, nhid, nclass, dropout, eeg_size):
        """
        :param nfeat: 输入的特征数量
        :param nhid: 隐藏层的特征数量
        :param nclass: 分类数
        :param eeg_size: 一个用于分界eeg信号和特征信息的数
        :param dropout:
        """
        super(GraphConvSTDP, self).__init__()
        self.eeg_size = eeg_size
        # 外部定义好图的样式
        self.adj_by_STDP = []

        self.adjs_constructor = GraphConstructSTDP(device="cuda:0")
        # 两层卷积层
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = dropout

    def forward(self, x):
        """
        :param x: [batch, epoch, channel, encoded eeg +features]
        :return:
        """

        with torch.no_grad():
            batch_size = x.shape[0]
            eeg = x[:, :, :self.eeg_size]
            features = x[:, :, self.eeg_size:]
            # print(f'eeg shape : {features.shape}')
            # print(f'x device : {x.device}')

            # todo 这里！！！！多进程的设置
            self.adj_by_STDP = self.adjs_constructor.get_batch_graph(input=eeg).to(torch.float32)
            # self.adj_by_STDP = adjs_constructor.get_batch_graph_multiprocessing(input=eeg).to(torch.float32)

        x = F.relu(self.gc1(features, self.adj_by_STDP))  # 第一层卷积
        x = F.dropout(x, self.dropout, training=self.training)  # 只在训练的时候启动
        x = self.gc2(x, self.adj_by_STDP)
        # print(x.shape)
        return F.log_softmax(x, dim=1)


class GraphConvStatic(nn.Module):
    """
    输入一个固定的图结构，并进行图卷积操作
    """

    def __init__(self, nfeat, nhid, nclass, dropout, adj):
        """
        :param nfeat: 输入的特征数量
        :param nhid: 隐藏层的特征数量
        :param nclass: 分类数
        :param dropout:
        """
        super(GraphConvStatic, self).__init__()
        # 外部定义好图的样式
        self.adj = adj

        # 两层卷积层
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = dropout

    def forward(self, x):
        batch_size = x.shape[0]
        # self.adj = self.adj.repeat(batch_size, 1, 1)
        x = F.relu(self.gc1(x, self.adj))  # 第一层卷积
        x = F.dropout(x, self.dropout, training=self.training)  # 只在训练的时候启动
        x = self.gc2(x, self.adj)
        # print(x.shape)
        return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    """

    定义GCN模型，即用预先定义的图卷积层来组建GCN模型。

    此部分与pytorch中构建经典NN模型的方法一致。

    """

    # GCN模型的输入是原始特征与图邻接矩阵，输出是结点最终的特征表示
    # 若对于一个包含图卷积的GCN来说，还需要指定隐层的维度。
    # 因此在GCN初始化的时候，有三个参数需要指定，输入层的维度，隐层的维度与输出层的维度。
    def __init__(self, nfeat, nhid, nclass, dropout):
        """
        :param nfeat: 输入的特征数量
        :param nhid: 隐藏层的特征数量
        :param nclass: 分类数 在cora任务中，这里是每个点有5个特征值，用来分类每个节点的
        :param dropout:
        """
        super(GCN, self).__init__()

        # 两层卷积层
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, adj):
        # print(f'x device{x.device}')
        # print(f'adj device{adj.device}')

        x = F.relu(self.gc1(x, adj))  # 第一层卷积
        x = F.dropout(x, self.dropout, training=self.training)  # 只在训练的时候启动
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    adj = torch.ones((2, 10, 10))
    net = nn.Sequential(
        # GraphConvStatic(nfeat=18, nhid=20, nclass=18, dropout=0.5, adj=adj), nn.ReLU(),
        # GraphConvSTDP(nfeat=18, nhid=20, nclass=18, dropout=0.5, eeg_size=1000), nn.ReLU(),
        # StdpGCN(nfeat=9, nhid=20, nclass=9, dropout=0.5, eeg_size=3000), nn.ReLU(),
        # StdpGCN_test(nfeat=9, nclass=9, dropout=0.5),
        StdpGCN_context(nfeat=9, nid=256, nclass=9, dropout=0.5),
        nn.Flatten(), nn.ReLU(),
        nn.Linear(90, 5)
    )

    # context
    input = torch.randn([2, 5, 10, 9])
    input_adj = torch.randn([2, 5, 10, 10])
    # no context
    # input = torch.randn([2, 10, 9])
    # input_adj = torch.randn([2, 10, 10])
    input = (input, input_adj)
    y = torch.randint(0, 4, [2])

    output = net(input)
    print(output.shape)
