"""
使用惊蛰框架的基类实现BSA编码器
"""

import spikingjelly
import scipy.io as scio
import numpy as np
import torch
from scipy import signal
from matplotlib import pyplot as plt
from scipy.signal import firwin
from sklearn import preprocessing
from tqdm import tqdm

from stdp_weight import STDP


class BSA:
    def __init__(self, threshold=0.955, filter_length=20, cutoff=0.8):
        self.threshold = threshold
        self.filter_length = filter_length

        filter_values = firwin(filter_length, cutoff=cutoff)
        self.filter = filter_values

    def encode(self, input):
        """
        输入一行信号，编码为同等长度的脉冲编码
        :param input: 单行信号
        :return: bsa编码脉冲信号
        """
        spike = [0]
        for i in range(1, len(input)):
            error1 = 0
            error2 = 0
            for j in range(1, self.filter_length):
                if i + j - 1 < len(input):
                    # print(f'i:{i} j:{j} i + j - 1:{i + j - 1} len filter {len(self.filter)}')
                    error1 += abs(input[i + j - 1] - self.filter[j])
                    error2 += abs(input[i + j - 1])

            if error1 <= error2 - self.threshold:
                # print(1)  # fire!!!
                spike.append(1)
                for j in range(1, self.filter_length):
                    if i + j - 1 < len(input):  # 这几个i+j-1可能不太对
                        input[i + j - 1] -= self.filter[j]
            else:
                # print(0)  # no fire
                spike.append(0)
        return np.array(spike)

    def multi_epoch_encode(self, input):
        """
        :param input: [epoch, channel, eeg]
        :return: [epoch, channel, spikes]
        """
        result = np.zeros((input.shape))
        epoch_num, channel_num, eeg_length = input.shape
        # TODO 这里应该可以写成多线程的
        pbar = tqdm(range(epoch_num))
        for epoch in pbar:
            pbar.set_description('encoding: ')
            for channel in range(channel_num):
                result[epoch][channel] = self.encode(input[epoch][channel])

        print(result.shape)
        return result

    def multi_channel_encode(self, input):
        pass

    def reconstruct(self, input):
        # TODO 计划是这里写把BSA编码反着重建原波形 但是不知道如何反向建立eeg波形的算法
        """
        计划是这里写把BSA编码反着重建原波形
        :param input: 输入一个
        :return:
        """
        pass


def norm_eeg(eeg_epoch_channel):
    Min = np.min(eeg_epoch_channel)
    Max = np.max(eeg_epoch_channel)
    after_norm = (eeg_epoch_channel - Min) / (Max - Min)
    return after_norm


def load_psg(path_extracted, subject_id, channels, resample=3000):
    """
    临时载入数据函数
    :param path_extracted:
    :param subject_id:
    :param channels:
    :param resample:重采样的样本点数
    :return:
    """
    psg = scio.loadmat('{}\\subject{}.mat'.format(path_extracted, subject_id))
    psg_resample = []
    for c in channels:
        psg_resample.append(
            np.expand_dims(psg[c], 1)  # 对原始信号扩展维度，在中间插入一个通道维度
        )
    psg_resample = np.concatenate(psg_resample, axis=1)  # 也就是把最后一个维度进行拼接，一个subject的数据为[epoch, 通道维， psg维]
    return psg_resample


def stdp_test():
    """
    这里是输入EEG信号并STDP
    :return:
    """
    path_extracted = 'D:\\data\\ISRUC_S3\\ExtractedChannels\\'
    channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1',
                'LOC_A2', 'ROC_A1', 'X1', 'X2']
    psg = load_psg(path_extracted, 1, channels)

    eeg_channel_0 = psg[521][0][:250]
    eeg_channel_1 = psg[521][3][:250]
    print(eeg_channel_0)

    #  将原始信号归一化
    after_norm_0 = norm_eeg(eeg_channel_0)
    after_norm_1 = norm_eeg(eeg_channel_1)

    # 使用BSA编码归一化后的eeg信号
    BSA_encoder = BSA()
    after_encode0 = torch.from_numpy(BSA_encoder.encode(after_norm_0)).view(-1, 1)
    after_encode1 = torch.from_numpy(BSA_encoder.encode(after_norm_1)).view(-1, 1)
    # print(after_encode0)

    # 使用迹实现STDP规则，STDP计算通道之间的权重
    stdp = STDP(pre_tau=100., post_tau=100.)
    stdp.get_trace_stdp_weight(pre_spikes=after_encode0, post_spikes=after_encode1)
    w = stdp.w

    # 画图
    fig = plt.figure(figsize=(11, 6))
    x = np.array(list(range(len(after_encode0))))
    T = len(after_encode0)
    after_encode0 = after_encode0[:, 0].numpy()
    after_encode1 = after_encode1[:, 0].numpy()

    # pre eeg and spike
    plt.subplot(3, 1, 1)
    plt.plot(x, after_norm_0)
    plt.eventplot((x * after_encode0)[after_encode0 == 1], lineoffsets=0.5, colors='r')
    plt.yticks([])
    plt.ylabel('$spike_{pre}$', rotation=0, labelpad=60, fontsize=18)
    plt.xticks([])
    plt.xlim(0, T)

    # post eeg and spike
    plt.subplot(3, 1, 2)
    plt.plot(x, after_norm_1)
    plt.eventplot((x * after_encode1)[after_encode1 == 1], lineoffsets=0.5, linelengths=2, colors='r')
    plt.yticks([])
    plt.ylabel('$spike_{post}$', rotation=0, labelpad=60, fontsize=18)
    plt.xticks([])
    plt.xlim(0, T)

    # w change
    plt.subplot(3, 1, 3)
    plt.plot(x, w)
    plt.ylabel('$w$', rotation=0, labelpad=30, fontsize=18)

    plt.show()
    fig.savefig('trace_stdp.pdf', format='pdf', bbox_inches='tight')



if __name__ == '__main__':
    # path_extracted = 'D:\\data\\ISRUC_S3\\ExtractedChannels\\'
    # channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1',
    #             'LOC_A2', 'ROC_A1', 'X1', 'X2']
    # psg = load_psg(path_extracted, 1, channels)
    #
    # eeg_channel_0 = psg[0:10][:][:500]
    # eeg_channel_1 = psg[0:10][:][:500]
    # print(eeg_channel_0.shape)
    #
    # #  将原始信号归一化
    # after_norm_0 = norm_eeg(eeg_channel_0)
    # after_norm_1 = norm_eeg(eeg_channel_1)
    #
    # # 使用BSA编码归一化后的eeg信号
    # BSA_encoder = BSA()
    # after_encode0 = torch.from_numpy(BSA_encoder.multi_epoch_encode(after_norm_0))
    # # after_encode1 = torch.from_numpy(BSA_encoder.multi_epoch_encode(after_norm_1))
    # print(after_encode0[0][0][200:230])
    # print(after_encode0[0][1][200:230])
    # print(after_encode0[0][2][200:230])
    stdp_test()



    # 使用迹实现STDP规则，STDP计算通道之间的权重
    # stdp = STDP(pre_tau=100., post_tau=100.)
    # stdp.get_trace_stdp_weight(pre_spikes=after_encode0, post_spikes=after_encode1)
    # w = stdp.w
    # # print(w)
