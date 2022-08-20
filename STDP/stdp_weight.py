import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer, neuron, functional
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from load_data import load_isruc_S3


class STDP:

    def __init__(self, pre_tau=100., post_tau=100.):
        """
        初始化参数， 并初始化属性，方便追踪数据的变化
        :param pre_spikes: 突触前脉冲序列
        :param post_spikes: 突触后脉冲序列
        :param pre_tau: 突触前tau参数
        :param post_tau:突触后tau参数
        """
        self.pre_spikes = []
        self.post_spikes = []
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.trace_pre = []
        self.trace_post = []
        self.w = []

    def get_trace_stdp_weight(self, pre_spikes, post_spikes, lr=1e-2):
        # 在这里可以首先对输入信号进行正则化
        self.pre_spikes = pre_spikes
        self.post_spikes = post_spikes

        fc = nn.Linear(1, 1, bias=False).to(pre_spikes.device)
        fc.weight.data = torch.tensor([[0.0]])  # 这里初始权重被初始化为0
        # print(f'weight init {fc.weight.item()}')

        stdp_learner = layer.STDPLearner(self.pre_tau, self.post_tau, self.f_pre, self.f_post)
        if len(self.pre_spikes) == len(self.post_spikes):
            T = len(self.pre_spikes)
            # for t in tqdm(range(T)):
            # print(self.pre_spikes.device)
            for t in range(T):
                stdp_learner.stdp(self.pre_spikes[t], self.post_spikes[t], fc, lr)
                # self.trace_pre.append(stdp_learner.trace_pre.item())
                # self.trace_post.append(stdp_learner.trace_post.item())
                self.w.append(fc.weight.item())
        else:
            raise NotImplementedError

        return fc.weight.item()

    def get_stdp_weight_multi(self, pre_spikes, post_spikes, channel_num=2, lr=1e-2, ):
        """
        使用多连接层 不需要两重for循环来构造 一个邻接矩阵
        :param pre_spikes: 突触前脉冲
        :param post_spikes: 突触后脉冲
        :param channel_num: eeg通道数量， 用来构造channel_num * channel_num的全连接层
        :param lr: 学习率
        :return:
        """
        self.pre_spikes = pre_spikes

        self.post_spikes = post_spikes

        # print(f'self.pre_spikes shape {self.pre_spikes.shape}')
        fc = nn.Linear(channel_num, channel_num, bias=False).to(pre_spikes.device)

        # print(f'weight init {fc.weight.data.shape}')
        fc.weight.data = torch.zeros_like(fc.weight.data)  # 这里初始权重被初始化为0
        # print(f'weight init {fc.weight.item()}')

        stdp_learner = layer.STDPLearner(self.pre_tau, self.post_tau, self.f_pre, self.f_post)
        if len(self.pre_spikes) == len(self.post_spikes):
            T = len(self.pre_spikes)

            for t in range(T):
                stdp_learner.stdp(self.pre_spikes[t], self.post_spikes[t], fc, lr)

        else:
            raise NotImplementedError

        return fc.weight.data

    # F+(wij),F−(wij)
    def f_pre(self, x):
        return x.abs() + 0.1

    def f_post(self, x):
        return - self.f_pre(x)


if __name__ == '__main__':
    # trace_pre = []
    # trace_post = []
    # w1 = []
    # w2 = []
    # T = 6000
    # channel_num = 2
    # # 数据构造
    # s_pre0 = torch.zeros([T, channel_num])
    # s_post = torch.zeros([T, channel_num])
    #
    # s_pre0[0: T // 2, 0] = (torch.rand_like(s_pre0[0: T // 2, 0]) > 0.95).float()
    # s_pre0[0: T // 2, 1] = (torch.rand_like(s_pre0[0: T // 2, 1]) < 0.95).float()
    #
    # s_post[0: T // 2, 0] = (torch.rand_like(s_post[0: T // 2, 0]) > 0.9).float()
    # s_post[0: T // 2, 1] = (torch.rand_like(s_post[0: T // 2, 1]) < 0.9).float()
    # s_post[T // 2:] = (torch.rand_like(s_post[T // 2:]) > 0.95).float()
    # print(s_post.shape)
    # # w1 = STDP().get_trace_stdp_weight(s_pre0, s_post)
    # # 单通道之间构造
    # w1 = STDP().get_stdp_weight_multi(s_pre0, s_post)
    # print(w1.t())
    #
    # stdp = STDP()
    #
    # w2 = stdp.get_trace_stdp_weight(s_pre0[:, 1].unsqueeze(1), s_post[:, 0].unsqueeze(1))
    # print(w2)

    train_iter, test_iter = load_isruc_S3(batch_size=20, path="D:\\data\\ISRUC_S3\\ISRUC_S3_features_origin.npz")

    input = []
    for x, y in train_iter:
        print(x.shape)
        print(y.shape)
        input = x[0, :, :3000]
        print(input.shape)
        break


    # x = torch.ones(10, 5)
    x = input
    channel_number, _ = x.shape
    # pre = torch.repeat_interleave(x, 10, dim=0)
    # post = x.repeat(10, 1)

    pre = x
    post = x

    stdp = STDP()
    print(x)
    print(f'pre shape : {pre.t().shape} post shape: {post.t().shape}')

    w_adj = stdp.get_stdp_weight_multi(pre.t(), post.t(), channel_num=channel_number)
    print(w_adj)


