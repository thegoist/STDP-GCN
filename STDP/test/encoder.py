"""
使用惊蛰框架的基类实现BSA编码器
"""

import scipy.io as scio
import numpy as np

from matplotlib import pyplot as plt
from scipy.signal import firwin
from tqdm import tqdm


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
        # TODO 这里可以写成多进程的
        for epoch in tqdm(range(epoch_num)):
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


if __name__ == '__main__':
    # path_extracted = 'D:\\data\\ISRUC_S3\\ExtractedChannels\\'
    # channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1',
    #             'LOC_A2', 'ROC_A1', 'X1', 'X2']
    # psg = load_psg(path_extracted, 1, channels)
    #
    # eeg_channel = psg[0:10][:][:500]

    # np.save('eeg_singe_10channel.npy', eeg_channel)

    # input -> [epoch, channel, eeg]
    eeg_channel = np.load('eeg_singe_10channel.npy')
    print(eeg_channel.shape)

    #  将原始信号归一化 第一个epoch 第一个channel
    single_channel = eeg_channel[0][0][:300]
    after_norm = norm_eeg(single_channel)
    print(after_norm.shape)

    # 使用BSA编码归一化后的eeg信号
    BSA_encoder = BSA()
    after_encode = BSA_encoder.encode(after_norm)

    # 画图
    fig = plt.figure(figsize=(11, 4))
    x = np.array(list(range(len(after_encode))))
    T = len(after_encode)

    # eeg and spike
    plt.subplot(1, 1, 1)
    plt.plot(x, after_norm)
    plt.eventplot((x * after_encode)[after_encode == 1], lineoffsets=0.5, colors='r')
    plt.yticks([])
    plt.ylabel('EEG $spikes$', rotation=0, labelpad=50, fontsize=18)
    plt.xticks([])
    plt.xlim(0, T)

    plt.show()

    fig.savefig('BSA.pdf', format='pdf', bbox_inches='tight')

