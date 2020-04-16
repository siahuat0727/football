import copy
import random
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_layers(dims):
    layers = []
    for idx, (dim_in, dim_out) in enumerate(zip(dims, dims[1:])):
        if idx != 0:
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dim_in, dim_out))
    return nn.Sequential(*layers)


class _FCNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = _make_layers(dims)

    def forward(self, x):
        return self.layers(x)


class _DuelingFCNet(nn.Module):
    def __init__(self, dims):
        super().__init__()

        *share_dims, last_dim = dims

        self.layers = _FCNet(share_dims)
        self.relu = nn.ReLU(inplace=True)

        self.net_v = _FCNet([share_dims[-1], 1])
        self.net_a = _FCNet([share_dims[-1], last_dim])

    def forward(self, x):
        mid = self.relu(self.layers(x))
        v = self.net_v(mid)
        a = self.net_a(mid)
        return v + (a - a.mean(dim=1, keepdim=True))  # TODO


class _QNet(nn.Module, ABC):
    def __init__(self, net):
        super().__init__()

        self._loss_f = nn.MSELoss()

        self.net = net
        self.net_t = None
        self.update_net_t()

    def update_net_t(self):
        self.net_t = copy.deepcopy(self.net).eval()
        for param in self.net_t.parameters():
            param.requires_grad = False

    def eps_greedy(self, s, n_a, eps=0.1, cuda=False):  # TODO not in eps
        if random.random() < eps:
            return random.randrange(n_a)
        else:
            if cuda:
                s = s.cuda()
            s = s.unsqueeze(0)
            return self.net_t(s).squeeze().argmax().item()

    def _q_eval(self, s, a):
        return self.net(s).gather(1, a)

    @abstractmethod
    def _q_next(self, s_):
        pass

    def _q_target(self, a, r, s_, done, gamma):
        mask = 1 - done.type(torch.uint8)
        return r + mask*gamma*self._q_next(s_)

    def _loss(self, batch, gamma):
        s, a, r, s_, done = batch
        q_eval = self._q_eval(s, a)
        q_target = self._q_target(a, r, s_, done, gamma)
        return self._loss_f(q_eval, q_target)

    def train(self, batch, optimizer, gamma=0.9):
        loss = self._loss(batch, gamma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


class _DQNMixin:
    def _q_next(self, s_):
        return self.net_t(s_).detach().max(dim=1, keepdim=True)[0]


class _DDQNMixin:
    def _q_next(self, s_):
        a = self.net(s_).argmax(dim=1, keepdim=True)
        return self.net_t(s_).gather(1, a)


class DQN(_DQNMixin, _QNet):
    def __init__(self, *args, **kwargs):
        net = _FCNet(*args, **kwargs)
        super().__init__(net)


class DDQN(_DDQNMixin, _QNet):
    def __init__(self, *args, **kwargs):
        net = _FCNet(*args, **kwargs)
        super().__init__(net)


class DuelingDQN(_DQNMixin, _QNet):
    def __init__(self, *args, **kwargs):
        net = _DuelingFCNet(*args, **kwargs)
        super().__init__(net)


class DuelingDDQN(_DDQNMixin, _QNet):
    def __init__(self, *args, **kwargs):
        net = _DuelingFCNet(*args, **kwargs)
        super().__init__(net)
