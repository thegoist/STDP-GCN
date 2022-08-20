import multiprocessing

import numpy as np
import torch

from stdp_weight import STDP


class GraphConstructorMulti:
    def __init__(self, adj_size):
        self.stdp = STDP()
        self.adj = np.zeros((adj_size))

    def get_graph(self, input, i):
        """
        单个图的生成
        用于并发
        :param input:
        :return:
        """

        channel_num, eeg_length = input.shape
        # adj = torch.eye(channel_num).float()

        pre = input
        post = input


        # print(pre.device)
        # print(f'pre shape : {pre.t().shape} post shape: {post.t().shape}')

        w_adj = self.stdp.get_stdp_weight_multi(pre.t(), post.t(), channel_num=channel_num)

        w_adj += np.identity(w_adj.shape[0])


        self.adj[i] = w_adj


def some_func(a, b):
    print(a + b)


if __name__ == '__main__':
    batch_size = 200

    eeg_test = torch.ones(batch_size, 10, 30)
    pool = multiprocessing.Pool(2)
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())

    graph_constructor = GraphConstructorMulti(adj_size=(batch_size, 10, 10))
    # graph_constructor.get_graph(eeg_test[0], i=0)
    # print(graph_constructor.adj)

    for i in range(batch_size):
        r = pool.apply_async(func=graph_constructor.get_graph, args=(eeg_test[i], i))
        # print(r.get())
    pool.close()
    print('-' * 100)
    pool.join()
    print(graph_constructor.adj)
