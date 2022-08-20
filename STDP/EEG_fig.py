
import numpy as np
import torch
from matplotlib import pyplot as plt

import scipy.io as scio

from encoder import norm_eeg
from stdp_weight import STDP


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



path_extracted = 'D:\\data\\ISRUC_S3\\ExtractedChannels\\'
channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1',
            'LOC_A2', 'ROC_A1', 'X1', 'X2']
psg = load_psg(path_extracted, 1, channels)


eegs = psg[521]
print(eegs.shape)

fig = plt.figure(figsize=(11, 6))
for i in range(10):
    test = eegs[i][:2000]
    # # 画图

    x = np.array(list(range(len(test))))
    test = norm_eeg(test) + i - 1
    T = len(test)
    # plt.subplot(11, 1, 1)
    plt.plot(x, test, color='cornflowerblue')
    plt.yticks([])
    plt.xticks([])
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.show()
fig.savefig('EEG.svg', format='svg', bbox_inches='tight')
#
# # pre eeg and spike
# plt.subplot(3, 1, 1)
# plt.plot(x, after_norm_0)
# plt.eventplot((x * after_encode0)[after_encode0 == 1], lineoffsets=0.5, colors='r')
# plt.yticks([])
# plt.ylabel('$spike_{pre}$', rotation=0, labelpad=60, fontsize=18)
# plt.xticks([])
# plt.xlim(0, T)
#
# # post eeg and spike
# plt.subplot(3, 1, 2)
# plt.plot(x, after_norm_1)
# plt.eventplot((x * after_encode1)[after_encode1 == 1], lineoffsets=0.5, linelengths=2, colors='r')
# plt.yticks([])
# plt.ylabel('$spike_{post}$', rotation=0, labelpad=60, fontsize=18)
# plt.xticks([])
# plt.xlim(0, T)
#
# # w change
# plt.subplot(3, 1, 3)
# plt.plot(x, w)
# plt.ylabel('$w$', rotation=0, labelpad=30, fontsize=18)
#
# plt.show()
# fig.savefig('EEG.pdf', format='pdf', bbox_inches='tight')