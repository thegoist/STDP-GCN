# 代码使用
## 数据
按顺序执行
- isruc_s3_preprocess.py : 
  - 预处理数据集 
  - 提取特征 然后编码 
  - 编码后使用stdp_graph.py中的方法生成邻接图 一并以npy的形式存入文件
- load_data.py : 
  - 读取数据，并对邻接图进行正则化，
  - 使用dataloader类构建训练和测试数据 
  - 注意在这里使用conext_add 将每个epoch都添加上下文的数据 _**[batch, context, channel, EEG]**_
- layer.py
  - 基础的网络层 一层GCN的实现
- models.py
  - 根据基础的网络层，构建的网络模型
  - 两层GCN
  - STDP版本的GCN
- train.py
  - 训练代码
  - 设置参数
  - 读取数据
  - 打印模型每层的形状
  - 调用训练代码
- utils.py
  - 训练函数
  - 工具函数

## 首先启动visdom 再启动训练`
> ```shell
> python -m visdom.server