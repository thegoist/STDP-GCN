import math

import numpy as np
import scipy.io as scio
from os import path

import torch
from scipy import signal
from scipy.fftpack import fft, ifft
from tqdm import tqdm
import torch.nn.functional as F

from STDP_graph import GraphConstructSTDP
from encoder import BSA
from load_data import add_context_single_sub

path_extracted = 'D:\\data\\ISRUC_S3\\ExtractedChannels\\'
path_raw = 'D:\\data\\ISRUC_S3\\RawData\\'
path_ouput = './data/ISRUC_S3/'

channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1',
            'LOC_A2', 'ROC_A1', 'X1', 'X2']


def norm_eeg(eeg_epoch_channel):
    Min = np.min(eeg_epoch_channel)
    Max = np.max(eeg_epoch_channel)
    after_norm = (eeg_epoch_channel - Min) / (Max - Min)
    return after_norm


def load_single_psg(path_extracted, subject_id, channels, resample=3000):
    psg = scio.loadmat('{}\\subject{}.mat'.format(path_extracted, subject_id))
    psg_resample = []
    for c in channels:
        psg_resample.append(
            np.expand_dims(signal.resample(psg[c], resample, axis=-1), 1)  # 对原始信号进行了重采样，扩展维度，在中间插入一个通道维度
            # np.expand_dims(psg[c], 1)  # 对原始信号进行了重采样，扩展维度，在中间插入一个通道维度
        )
    psg_resample = np.concatenate(psg_resample, axis=1)  # 也就是把最后一个维度进行拼接，一个subject的数据为[epoch, 通道维， psg维]
    return psg_resample


def load_single_stage_labels(path_raw, subject_id, ignore=30):
    # 读数据，因为重采样，必然会减少序列长度，因此stage后几个缺少 这里似乎不对？ 不理解为何去掉后30个 发现是多了30个标签点，不清楚为何多了30个标签点
    labels = np.genfromtxt('{}\\{}\\{}_1.txt'.format(path_raw, subject_id, subject_id), dtype=np.int32)
    return labels[:-ignore]


def get_single_de_psd(psg):
    """

    :param psg:
    :return:
    """
    stftn = 7680  # 不懂为什么这样设置 是指EEG的频域应该
    fStart = [0.5, 2, 4, 6, 8, 11, 14, 22, 31]  # 是每个频域 阿尔法波什么的
    fEnd = [4, 6, 8, 11, 14, 22, 31, 40, 50]  # 是每个频域 阿尔法波什么的
    window = 30  # 窗口长度 30秒 todo 这是原代码的超参数
    # window = 10  # 窗口长度 10秒
    fs = 100  # 采样频率
    # fs = 200  # 采样频率

    epochs = psg.shape[0]
    channel_num = psg.shape[1]  # 拿到channel的数量
    DE = np.zeros([psg.shape[0], channel_num, len(fStart)])
    PSD = np.zeros([psg.shape[0], channel_num, len(fStart)])

    window_points = fs * window
    hanning_window = np.array(
        [0.5 - 0.5 * np.cos(2 * np.pi * n / (window_points + 1)) for n in range(1, window_points + 1)])

    fStartNum = np.zeros([len(fStart)], dtype=int)
    fEndNum = np.zeros([len(fEnd)], dtype=int)
    for i in range(0, len(fStart)):
        fStartNum[i] = int(fStart[i] / fs * stftn)
        fEndNum[i] = int(fEnd[i] / fs * stftn)

    # print(f'epoch {epochs} channel{channel_num}')
    # PSD = np.zeros([psg.shape[0], psg.shape[0], psg.shape[0]], dtype=int)
    for epoch in range(epochs):

        for channel in range(channel_num):
            channel_data = psg[epoch][channel]
            # print(f'channel data shape {channel_data.shape} hanning_window {hanning_window.shape}')
            after_window = channel_data * hanning_window
            FFT = fft(after_window, stftn)
            magFFT = abs(FFT[0:int(stftn / 2)])

            for p in range(len(fStart)):
                E = 0
                # E_log = 0
                for p0 in range(fStartNum[p] - 1, fEndNum[p]):
                    E = E + magFFT[p0] * magFFT[p0]
                E = E / (fEndNum[p] - fStartNum[p] + 1)

                PSD[epoch][channel][p] = E
                DE[epoch][channel][p] = math.log(100 * E, 2)

    # result = np.concatenate([PSD, DE], axis=-1)
    result = DE  # only de
    print(f'single de shape: {result.shape}')
    # return psg
    return result


def preprocess_features():
    labels = []
    psgs = []
    lens = []
    for sub in range(1, 11):
        print('Reading subject', sub)
        label = load_single_stage_labels(path_raw, sub)
        psg = load_single_psg(path_extracted, sub, channels)
        print('Subject', sub, ':', label.shape, psg.shape)
        assert len(label) == len(psg)
        label[label == 5] = 4  # 标签为5的实际是上REM阶段，这里将标记换为4  0 1 2 3 4

        print(psg.shape)
        # print(f'after psg shape {get_single_de_psd(psg).shape}')
        # labels.append(np.eye(5)[label])  # 编码为one hot编码 添加到总的里面
        labels.append(label)  # 编码为one hot编码 添加到总的里面
        # psgs.append(psg)
        psgs.append(get_single_de_psd(psg))
        lens.append(len(label))
    # print(f'final psgs_features.shape {np.array(psgs).shape}')

    np.savez('D:\\data\\ISRUC_S3\\ISRUC_S3.npz',
             psg=psgs,
             labels=labels,
             lens=lens
             )

    print("preprocess done!")


def preprocess_eeg_features_adj():
    """
    首先数据预处理的时候，将eeg数据进行归一化和编码
    :return:
    """
    labels = []
    psgs_features = []
    subject_adjs = []
    lens = []
    for sub in range(1, 11):
        print('Reading subject', sub)
        # 读入数据 和 处理标签值
        label = load_single_stage_labels(path_raw, sub)
        psg = load_single_psg(path_extracted, sub, channels)
        print('Subject', sub, ':', 'label shape:', label.shape, 'psg shape: ', psg.shape)
        assert len(label) == len(psg)
        label[label == 5] = 4  # 标签为5的实际是上REM阶段，这里将标记换为4  0 1 2 3 4

        # print(f'after psg shape {get_single_de_psd(psg).shape}')
        # labels.append(np.eye(5)[label])  # 编码为one hot编码 添加到总的里面
        labels.append(label)  # 编码为one hot编码 添加到总的里面
        lens.append(len(label))

        # 归一化EEG 然后编码
        psg = norm_eeg(psg)
        BSA_encoder = BSA()
        after_encode_psg = BSA_encoder.multi_epoch_encode(psg)
        print(f'encode psg shape :{after_encode_psg.shape}')
        psgs_features_concat = np.concatenate([after_encode_psg, get_single_de_psd(psg)], axis=-1)
        psgs_features.append(psgs_features_concat)
        print(f'psgs_features_concat.shape {psgs_features_concat.shape}')

        # 构建邻接图并存下来
        stdp_adj_constructor = GraphConstructSTDP(device="cuda:0")
        subject_adj = stdp_adj_constructor.get_batch_graph_test(torch.from_numpy(after_encode_psg).to("cuda:0"))
        subject_adjs.append(subject_adj)
        print(f'subject_adj.shape {subject_adj.shape}')

    print(f'final psgs_features.shape {np.array(psgs_features).shape}')
    print(f'final adjs.shape {np.array(subject_adjs).shape}')

    np.savez('D:\\data\\ISRUC_S3\\ISRUC_S3_features_origin.npz',
             psg=psgs_features,
             # psg_features=psgs_features,
             subject_adjs=subject_adjs,
             labels=labels,
             lens=lens,
             )

    print("preprocess done!")


def preprocess_features_adj():
    """
    首先数据预处理的时候，将eeg数据进行归一化和编码
    只存储提取的特征和邻接图
    # 邻接图未经过了归一化
    :return:
    """
    labels = []
    psgs_features = []
    subject_adjs = []
    lens = []
    for sub in range(1, 11):
        print('Reading subject', sub)
        # 读入数据 和 处理标签值
        label = load_single_stage_labels(path_raw, sub)
        psg = load_single_psg(path_extracted, sub, channels)
        print('Subject', sub, ':', 'label shape:', label.shape, 'psg shape: ', psg.shape)
        assert len(label) == len(psg)
        label[label == 5] = 4  # 标签为5的实际是上REM阶段，这里将标记换为4  0 1 2 3 4

        # print(f'after psg shape {get_single_de_psd(psg).shape}')
        # labels.append(np.eye(5)[label])  # 编码为one hot编码 添加到总的里面
        labels.append(label)  # 编码为one hot编码 添加到总的里面
        lens.append(len(label))

        # 归一化EEG 然后编码
        psg = norm_eeg(psg)
        BSA_encoder = BSA()
        after_encode_psg = BSA_encoder.multi_epoch_encode(psg)
        print(f'encode psg shape :{after_encode_psg.shape}')
        features = get_single_de_psd(psg)
        psgs_features.append(features)
        print(f'features.shape {features.shape}')

        # 构建邻接图并存下来
        stdp_adj_constructor = GraphConstructSTDP(device="cuda:0")
        subject_adj = stdp_adj_constructor.get_batch_graph_test(torch.from_numpy(after_encode_psg).to("cuda:0"))
        subject_adjs.append(subject_adj)
        print(f'subject_adj.shape {subject_adj.shape}')

    print(f'final psgs_features.shape {np.array(psgs_features).shape}')
    print(f'final adjs.shape {np.array(subject_adjs).shape}')

    np.savez('D:\\data\\ISRUC_S3\\ISRUC_S3_features_adjs_re.npz',
             psg=psgs_features,
             # psg_features=psgs_features,
             subject_adjs=subject_adjs,
             labels=labels,
             lens=lens,
             )

    print("preprocess done!")


def delete_eeg():
    """
    主要是将原有的数据里把EEG去掉
    添加上下文

    :return:
    """
    read = np.load("D:\\data\\ISRUC_S3\\ISRUC_S3_features_origin.npz", allow_pickle=True)

    psgs = read['psg']
    labels = read['labels']
    lens = read['lens']
    adjs = read['subject_adjs']
    print(f'psg shape {type(psgs)}')
    psgs_list = []
    for sub in psgs:
        psgs_list.append(sub[:, :, 3009:])
        print(f'psg shape {sub.shape}')
    psgs = np.array(psgs_list)

    np.savez('D:\\data\\ISRUC_S3\\ISRUC_S3_features_adjs.npz',
             psg=psgs,
             subject_adjs=adjs,
             labels=labels,
             lens=lens,
             )

    print("preprocess done!")


if __name__ == '__main__':
    # preprocess_features()
    # preprocess_eeg_features_adj()
    preprocess_features_adj()
    # delete_eeg()
