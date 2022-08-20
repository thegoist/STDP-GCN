import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer, neuron, functional
from matplotlib import pyplot as plt
import numpy as np


# F+(wij),F−(wij)
def f_pre(x):
    return x.abs() + 0.1


def f_post(x):
    return - f_pre(x)


fc1 = nn.Linear(1, 1, bias=False)
fc2 = nn.Linear(1, 1, bias=False)

stdp_learner = layer.STDPLearner(100., 100., f_pre, f_post)
trace_pre = []
trace_post = []
w1 = []
w2 = []
T = 256
print(T // 2)
s_pre0 = torch.zeros([T, 1])
s_pre1 = torch.zeros([T, 1])
s_post = torch.zeros([T, 1])
s_pre0[0: T // 2] = (torch.rand_like(s_pre0[0: T // 2]) > 0.95).float()
s_pre1[0: T // 2] = (torch.rand_like(s_pre0[0: T // 2]) > 0.5).float()
s_post[0: T // 2] = (torch.rand_like(s_post[0: T // 2]) > 0.9).float()

s_pre0[T // 2:] = (torch.rand_like(s_pre0[T // 2:]) > 0.8).float()
s_pre1[T // 2:] = (torch.rand_like(s_pre0[T // 2:]) > 0.99).float()
s_post[T // 2:] = (torch.rand_like(s_post[T // 2:]) > 0.95).float()
print(s_pre0.shape)
print(s_post.shape)
for t in range(T):
    stdp_learner.stdp(s_pre0[t], s_post[t], fc1, 1e-2)
    stdp_learner.stdp(s_pre1[t], s_post[t], fc2, 1e-2)
    trace_pre.append(stdp_learner.trace_pre.item())
    trace_post.append(stdp_learner.trace_post.item())
    w1.append(fc1.weight.item())
    w2.append(fc2.weight.item())

# plt.style.use('science')
fig = plt.figure(figsize=(11, 6))
s_pre0 = s_pre0[:, 0].numpy()
s_post = s_post[:, 0].numpy()
t = np.arange(0, T)
plt.subplot(6, 1, 1)
plt.eventplot((t * s_pre0)[s_pre0 == 1.], lineoffsets=0, colors='r')
plt.yticks([])
plt.ylabel('$S_{pre}$', rotation=0, labelpad=10, )
plt.xticks([])
plt.xlim(0, T)
plt.subplot(6, 1, 2)
plt.plot(t, trace_pre)
plt.ylabel('$tr_{pre}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.subplot(6, 1, 3)
plt.eventplot((t * s_post)[s_post == 1.], lineoffsets=0, colors='r')
plt.yticks([])
plt.ylabel('$S_{post}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)
plt.subplot(6, 1, 4)
plt.plot(t, trace_post)
plt.ylabel('$tr_{post}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)
plt.subplot(6, 1, 5)
plt.plot(t, w1)
plt.ylabel('$w1$', rotation=0, labelpad=10)
plt.xlim(0, T)

plt.subplot(6, 1, 6)
plt.plot(t, w2)
plt.ylabel('$w2$', rotation=0, labelpad=10)
plt.xlim(0, T)

plt.show()
