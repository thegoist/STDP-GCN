"""
以subject为单位划分数据
1.输入为  [个体， 个体记录的epochs, 10通道， 脑电序列]
        [个体, epochs对应的状态lable(长度5的list one hot)]
        batch size
2. 输出为 data_iter, label_iter
    data_iter = [batch_size的epoch数量, 10通道, 脑电序列]
    label_iter = [batch_size的epoch数量, 5维度的one hot变量（标签）]

编程思路：
    1 拼接所有人的epochs，对应的标签也进行拼接
    2 应用yield进行输出

"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
import torch.nn.functional as F
from tqdm import tqdm


def add_context_single_sub(x, y, adj, context):
    """
    input:
        x       : 输入的训练eeg [batch,  channel, eeg]
        y       : 标签数据 [batch,  label]

        context : 上下文的数量 int

    return:
        x with contexts. [batch, context, channel, eeg]
        y with contexts. [batch, context, label]
    """
    # print(f'x shape {x.shape}')
    # print(f'y shape {y.shape}')
    if context != 1:
        cut = context // 2
    else:
        return np.expand_dims(x, 1),y ,adj.unsqueeze(1).cpu()

    context_x = np.zeros([x.shape[0] - 2 * cut, context, x.shape[1], x.shape[2]], dtype=float)
    context_adj = torch.zeros(x.shape[0] - 2 * cut, context, adj.shape[1], adj.shape[2])
    for i in range(cut, x.shape[0] - cut):
        context_x[i - cut] = x[i - cut:i + cut + 1]
        context_adj[i - cut] = adj[i - cut:i + cut + 1]

    context_y = y[cut:-cut]

    # print(f'x_c shape {context_x.shape}')
    # print(f'y_c shape {context_y.shape}')
    # print(f'context_adj shape {context_adj.shape}')

    return context_x, context_y, context_adj


def load_data_isruc_k_fold(psgs_concat, adjs_concat, labels_concat, train, K, i):
    """
    输入K I ，返回K折下第I个数据划分
    :param psgs_concat:
    :param adjs_concat:
    :param labels_concat:
    :param train:
    :param K:
    :param i:
    :return:
    """
    assert K > 1
    fold_size = psgs_concat.shape[0] // K  # 每份的个数:数据总条数/折数（组数）

    num_examples = len(psgs_concat)
    # print(f'examples number: {num_examples}')
    # print(f'adjs_concat: {adjs_concat.shape}')
    # print(f'features_concat: {psgs_concat.shape}')
    # print(f'label: {labels_concat.shape}')

    psgs_train, adjs_train, labels_train = None, None, None
    psgs_test, adjs_test, labels_test = None, None, None

    for j in range(K):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数

        # idx 为每组 valid
        psgs_part, adjs_part, labels_part = psgs_concat[idx], adjs_concat[idx], labels_concat[idx]
        if j == i:  # 第i折 为 test data
            # print(idx)
            psgs_test, adjs_test, labels_test = psgs_part, adjs_part, labels_part
        elif psgs_train is None:
            psgs_train, adjs_train, labels_train = psgs_part, adjs_part, labels_part
        else:
            psgs_train = np.concatenate((psgs_train, psgs_part))  # dim=0增加行数，竖着连接
            adjs_train = np.concatenate((adjs_train, adjs_part))
            labels_train = np.concatenate((labels_train, labels_part))

    if train:
        return psgs_train, adjs_train, labels_train
    else:
        return psgs_test, adjs_test, labels_test


class ISRUC_S3__subject_k_fold(Dataset):
    def __init__(self, data_path, num_context, K, i, train=True):
        super(ISRUC_S3__subject_k_fold, self).__init__()
        if train:
            self.trains, self.adjs, self.labels = get_subject_k_fold_isruc_S3(data_path, False, num_context, train=train, K=K, i=i)
        else:
            self.trains, self.adjs, self.labels = get_subject_k_fold_isruc_S3(data_path, False, num_context, train=train, K=K, i=i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.trains[index], self.adjs[index], self.labels[index]


def get_subject_k_fold_isruc_S3(path, shuffle, num_context, train, K, i):
    """这里直接K折划分数据。"""
    read = np.load(path, allow_pickle=True)
    psgs = read['psg']
    labels = read['labels']
    adjs = read['subject_adjs']

    psgs_train, adjs_train, labels_train = None, None, None
    psgs_test, adjs_test, labels_test = None, None, None

    # 按照subject进行k折划分
    for sub in range(K):
        # 添加上下文
        psgs_context, labels_context, adjs_context = add_context_single_sub(psgs[sub], labels[sub], adjs[sub],
                                                                            context=num_context)


        psgs_part,  labels_part, adjs_part, = psgs_context, labels_context, adjs_context



        # mean = np.mean(psgs_part, 3)
        # psgs_part = psgs_part - mean.reshape((mean.shape[0],mean.shape[1],mean.shape[2], -1))

        if sub == i:  # 第i折 为 test data
            # print(idx)

            psgs_test, adjs_test, labels_test = psgs_part, adjs_part, labels_part
            # psgs_test, adjs_test, labels_test = psgs_part, adjs_part, labels_part
        elif psgs_train is None:
            psgs_train, adjs_train, labels_train = psgs_part, adjs_part, labels_part
        else:
            psgs_train = np.concatenate((psgs_train, psgs_part))  # dim=0增加行数，竖着连接
            adjs_train = torch.cat([adjs_train, adjs_part])
            labels_train = np.concatenate((labels_train, labels_part))


    # 这里预处理一下train test
    # adjs_concat = F.normalize(adjs_concat, p=1, dim=1)
    adjs_test = adjs_test != 0.  # 凡是有连接的都是1
    # adjs_concat = adjs_concat > 0  # 连接权值大于0的是1
    adjs_test = adjs_test.float()
    adjs_test += torch.eye(10)

    adjs_train = adjs_train != 0.  # 凡是有连接的都是1
    # adjs_concat = adjs_concat > 0  # 连接权值大于0的是1
    adjs_train = adjs_train.float()
    adjs_train += torch.eye(10)


    # # 在这里打乱数据 是否打乱顺序需要思考
    num_examples = len(psgs_train)
    if shuffle:
        indexs = np.arange(num_examples)

        np.random.shuffle(indexs)
        psgs_train = psgs_train[indexs]
        adjs_train = adjs_train[indexs]
        labels_train = labels_train[indexs]

    num_examples = len(psgs_test)
    if shuffle:
        indexs = np.arange(num_examples)

        np.random.shuffle(indexs)
        psgs_test = psgs_test[indexs]
        adjs_test = adjs_test[indexs]
        labels_test = labels_test[indexs]

    if train:
        return psgs_train, adjs_train, labels_train
    else:
        return psgs_test, adjs_test, labels_test



########k折划分############
def get_subject_k_fold_data(k, i, batch_size, data_path, num_context):
    isruc_train = ISRUC_S3__subject_k_fold(data_path, num_context, k, i, train=True)
    isruc_test = ISRUC_S3__subject_k_fold(data_path, num_context, k, i, train=False)

    return (
        DataLoader(isruc_train, batch_size, shuffle=False, num_workers=0),
        DataLoader(isruc_test, batch_size, shuffle=False, num_workers=0)
    )




if __name__ == '__main__':
    # train, test = load_isruc_S3(50, path="D:\\data\\ISRUC_S3\\ISRUC_S3.npz")  # 不包含eeg源数据的数据
    # train, test = load_isruc_S3(50, path="D:\\data\\ISRUC_S3\\ISRUC_S3_features_origin.npz")  # eeg源数据和预先算好的
    # train, test = load_isruc_S3(50, path="D:\\data\\ISRUC_S3\\ISRUC_S3_features_adjs_re.npz",
    #                             shuffle=True)  # eeg源数据和预先算好的
    # for i, (x, adjs, y) in enumerate(train):
    #     print(adjs[0][0])
    #     # break
    # train, test = load_isruc_S3_k_fold(50, path="D:\\data\\ISRUC_S3\\ISRUC_S3_features_adjs_re.npz", shuffle=False)
    k = 10
    # psgs_concat, adjs_concat, labels_concat = load_all_isruc_S3(
    #     path="D:\\data\\ISRUC_S3\\ISRUC_S3_features_adjs_re.npz", shuffle=False)
    for i in range(k):
        # data = get_k_fold_data(k, i, X_train, y_train)  # 获取k折交叉验证的训练和验证数据
        # train_iter, test_iter = get_subject_k_fold_data(data_path="G:\\zhaoyuan\\ISRUC_S3_\\ISRUC_S3\\ISRUC_S3_features_adjs_re.npz",
        #                                                 k=k, i=i, batch_size=50,num_context=5)
        train_iter, test_iter = get_subject_k_fold_data(data_path="G:\\zhaoyuan\\ISRUC_S3_\\ISRUC_S3\\ISRUC_S3_features_adjs_re_FS200.npz",
                                                        k=k, i=i, batch_size=50,num_context=5)
        print(i)
