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


fc = nn.Linear(1, 1, bias=False)

stdp_learner = layer.STDPLearner(100., 100., f_pre, f_post)
trace_pre = []
trace_post = []
w = []
T = 256
s_pre = torch.zeros([T, 1])
s_post = torch.zeros([T, 1])
s_pre[0: T // 2] = (torch.rand_like(s_pre[0: T // 2]) > 0.95).float()
s_post[0: T // 2] = (torch.rand_like(s_post[0: T // 2]) > 0.9).float()

s_pre[T // 2:] = (torch.rand_like(s_pre[T // 2:]) > 0.8).float()
s_post[T // 2:] = (torch.rand_like(s_post[T // 2:]) > 0.95).float()

for t in range(T):
    stdp_learner.stdp(s_pre[t], s_post[t], fc, 1e-2)
    trace_pre.append(stdp_learner.trace_pre.item())
    trace_post.append(stdp_learner.trace_post.item())
    w.append(fc.weight.item())

# plt.style.use('science')
fig = plt.figure(figsize=(10, 6))
s_pre = s_pre[:, 0].numpy()
s_post = s_post[:, 0].numpy()
t = np.arange(0, T)
plt.subplot(5, 1, 1)

plt.eventplot((t * s_pre)[s_pre == 1.], lineoffsets=0, colors='r')
plt.yticks([])
plt.ylabel('$S_{pre}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)
plt.subplot(5, 1, 2)
plt.plot(t, trace_pre)
plt.ylabel('$tr_{pre}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.subplot(5, 1, 3)
plt.eventplot((t * s_post)[s_post == 1.], lineoffsets=0, colors='r')
plt.yticks([])
plt.ylabel('$S_{post}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)
plt.subplot(5, 1, 4)
plt.plot(t, trace_post)
plt.ylabel('$tr_{post}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)
plt.subplot(5, 1, 5)
plt.plot(t, w)
plt.ylabel('$w$', rotation=0, labelpad=10)
plt.xlim(0, T)

plt.show()
