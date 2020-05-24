import copy
import random
from functools import partial
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.distributions import Categorical


def _make_layers(dims, activation=nn.ReLU(inplace=True)):
    layers = []
    for idx, (dim_in, dim_out) in enumerate(zip(dims, dims[1:])):
        if idx != 0:
            layers.append(activation)
        layers.append(nn.Linear(dim_in, dim_out))
    return nn.Sequential(*layers)


def _onehot(v, n):
    ret = torch.eye(n)[v.squeeze()].to(v.device)
    assert v.dim() == ret.dim(), f'{v.size()} {ret.size()}'
    return ret


class _MLPNet(nn.Module):

    def __init__(self, in_dim, hid_dims, out_dim, activation=nn.ReLU(), last_layer=None, **kwargs):
        super().__init__()
        self.layers = _make_layers([in_dim, *hid_dims, out_dim], activation)
        self.last_layer = last_layer

    def forward(self, x):
        x = self.layers(x)
        if self.last_layer is not None:
            x = self.last_layer(x)
        return x


class _DuelingNet(nn.Module):
    def __init__(self, in_dim, hid_dims, out_dim, **kwargs):
        super().__init__()

        self.layers = _make_layers([in_dim, *hid_dims])
        self.relu = nn.ReLU(inplace=True)

        self.net_v = _make_layers([hid_dims[-1], 1])
        self.net_a = _make_layers([hid_dims[-1], out_dim])

    def forward(self, x):
        mid = self.relu(self.layers(x))
        v = self.net_v(mid)
        a = self.net_a(mid)
        return v + (a - a.mean(dim=1, keepdim=True))


class _Net(nn.Module, ABC):
    def __init__(self, net, lr=1e-3, **kwargs):
        super().__init__()
        self.net = net
        self.optim = Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)

    @abstractmethod
    def _loss(self, batch, *args, **kwargs):
        pass

    def step(self, batch, **kwargs):
        loss = self._loss(batch, **kwargs)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optim.step()
        return loss.item()


class _DoubleNetMixin:
    def __init__(self, *args, update_mode='hard', **kwargs):
        super().__init__(*args, **kwargs)

        self.n_step = 0
        self.update_freq = kwargs.pop('update_freq')
        assert self.update_freq > 0, f'Invalid freq {update_freq}'

        self.net_t = None
        self.hard_update()

        self.update = {
            'soft': self.soft_update,
            'hard': self.hard_update,
        }.get(update_mode, None)
        assert self.update is not None, f'Unknown mode {update_mode}'

    def hard_update(self, **kwargs):
        self.net_t = copy.deepcopy(self.net)

    def soft_update(self, tau=0.01, **kwargs):
        for src, dst in zip(self.net.parameters(), self.net_t.parameters()):
            dst.data.copy_(dst.data * (1.0 - tau) + src.data * tau)

    def step(self, *args, **kwargs):
        self.n_step += 1

        ret = super().step(*args, **kwargs)

        if self.n_step == self.update_freq:
            self.update(**kwargs)
            self.n_step = 0

        return ret


class _QNet(_Net):
    def __init__(self, *args, **kwargs):  # TODO update
        super().__init__(*args, **kwargs)

        self._loss_fn = nn.MSELoss()

    def choose_act(self, s, **kwargs):
        return self.net_t(s).squeeze().argmax(), None

    def _q_target(self, r, s_, gamma, **kwargs):
        return r + gamma*self._q_next(s_, **kwargs)

    def _loss(self, batch, gamma=0.999, **kwargs):
        q_eval = self._q_eval(batch.s, batch.a, **kwargs)
        q_target = self._q_target(batch.r, batch.s_, gamma, **kwargs)
        return self._loss_fn(q_eval, q_target)


class _QLearningMixin:
    def _q_eval(self, s, a):
        return self.net(s).gather(1, a)

    def _q_next(self, s_):
        return self.net_t(s_).detach().max(dim=1, keepdim=True)[0]


class _DQLearningMixin(_QLearningMixin):
    def _q_next(self, s_):
        a = self.net(s_).argmax(dim=1, keepdim=True).detach()
        return self.net_t(s_).gather(1, a)


class DQN(_QNet, _QLearningMixin):
    def __init__(self, *args, **kwargs):
        _net = _MLPNet(*args, **kwargs)
        super().__init__(_net, *args, **kwargs)


class DDQN(_QNet, _DQLearningMixin):
    def __init__(self, *args, **kwargs):
        _net = _MLPNet(*args, **kwargs)
        super().__init__(_net, *args, **kwargs)


class DuelingDQN(_QNet, _QLearningMixin):
    def __init__(self, *args, **kwargs):
        _net = _DuelingNet(*args, **kwargs)
        super().__init__(_net, *args, **kwargs)


class DuelingDDQN(_QNet, _DQLearningMixin):
    def __init__(self, *args, **kwargs):
        _net = _DuelingNet(*args, **kwargs)
        super().__init__(_net, *args, **kwargs)


class ActorCritic(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.actor = _MLPNet(last_layer=nn.Softmax(dim=-1), **kwargs)
        kwargs['out_dim'] = 1
        self.critic = _MLPNet(**kwargs)


class PPO(_Net):
    def __init__(self, reward_normalize=False, **kwargs):
        _net = ActorCritic(activation=nn.Tanh(), **kwargs)
        super().__init__(_net, **kwargs)

        self.mse_loss = nn.MSELoss(reduction='none')
        self.clip_eps = 0.2
        self.reward_normalize = reward_normalize

    def _s2d(self, s):
        return Categorical((self.net.actor(s)))

    def _a2prob(self, a, dist):
        a_ = a.squeeze()
        assert a_.dim() <= 1, f'size = {a.size()}'
        return dist.log_prob(a_).view(a.size())

    def choose_act(self, s, **kwargs):
        with torch.no_grad():
            dist = self._s2d(s)
        a = dist.sample()

        logprobs = self._a2prob(a, dist)
        return a, logprobs

    def analysis(self, s, a):
        dist = self._s2d(s)
        logprobs = self._a2prob(a, dist)
        return logprobs, dist.entropy().view(a.size())

    def _loss(self, batch, **kwargs):

        r = batch.r
        if self.reward_normalize:
            r = (r - r.mean()) / (r.std() + 1e-20)

        logprobs, entropy = self.analysis(batch.s, batch.a)
        assert logprobs.size() == batch.logprobs.size(), \
            f'{logprobs.size()}, {batch.logprobs.size()}'

        v = self.net.critic(batch.s)

        # advantage A(s,a) = R + yV(s') - V(s)
        advs = r - v.detach()

        ratios = torch.exp(logprobs - batch.logprobs)
        actor_loss1 = ratios * advs
        actor_loss2 = torch.clamp(
            ratios, 1-self.clip_eps, 1+self.clip_eps) * advs

        actor_loss = torch.min(actor_loss1, actor_loss2)
        critic_loss = 0.5 * self.mse_loss(v, r)
        entropy_loss = 0.001 * entropy

        loss = critic_loss - actor_loss - entropy_loss
        return loss.mean()


class Actor(_DoubleNetMixin, _Net):
    def __init__(self, discrete=True, **kwargs):
        _net = _MLPNet(**kwargs)
        assert kwargs['update_freq'] == 1
        super().__init__(_net, update_mode='soft', **kwargs)

        if discrete:
            self.softmax = partial(
                torch.nn.functional.gumbel_softmax, hard=True)
        else:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, *args, add_noise=False, use_target=False, **kwargs):
        net = self.net_t if use_target else self.net
        x = net(*args, **kwargs)

        if add_noise:
            x += (torch.randn(x.size()).to(x.device) * 0.1).clamp(-1, 1)

        return self.softmax(x)

    def _loss(self, batch, critic=None, **kwargs):
        act_logits = self.forward(batch.s, add_noise=True)

        # TODO regularize?+ 1e-3*(act_logits**2).mean()  #TODO l2norm
        loss = -critic(batch.s, act_logits).mean()
        return loss


class Critic(_DoubleNetMixin, _QNet):
    def __init__(self, **kwargs):
        _net = _MLPNet(**kwargs)
        assert kwargs['update_freq'] == 1
        super().__init__(_net, update_mode='soft', **kwargs)
        self.n_a = kwargs['n_a']

    def forward(self, s, a=None, use_target=False):
        if a is None:
            input_ = s
        else:
            input_ = torch.cat([s, a], dim=1)
        net = self.net_t if use_target else self.net
        return net(input_)

    def _q_eval(self, s, a, **kwargs):
        return self.forward(s, _onehot(a, self.n_a))

    def _q_next(self, s_, actor=None, **kwargs):
        return self.forward(s_, actor(s_, use_target=True), use_target=True).detach()


class DDPG(nn.Module):

    def __init__(self, *args, tau=0.01, activation=nn.Tanh(), critic=Critic, **kwargs):
        super().__init__()

        self.tau = tau
        kwargs['activation'] = activation
        self.actor = Actor(*args, **kwargs)  # TODO lra, lrc

        n_a = kwargs.get('out_dim')
        kwargs['in_dim'] += kwargs['out_dim']
        kwargs['out_dim'] = 1
        self.critic = critic(*args, n_a=n_a, **kwargs)

    def choose_act(self, s, add_noise=True):
        return self.actor(s, add_noise=add_noise, use_target=True).argmax(), None

    def _loss(self):
        raise NotImplemented

    def step(self, batch, **kwargs):
        critic_loss = self.critic.step(batch, actor=self.actor, tau=self.tau)
        actor_loss = self.actor.step(batch, critic=self.critic, tau=self.tau)

        return critic_loss, actor_loss


class DoubleCritic:
    def __init__(self, *args, **kwargs):
        self.net1 = Critic(*args, **kwargs)
        self.net2 = Critic(*args, **kwargs)

        self._q_next1, self.net1._q_next = self.net1._q_next, self._min_q_next
        self._q_next2, self.net2._q_next = self.net2._q_next, self._min_q_next
        self._q_next_cache = None

    def _min_q_next(self, *args, **kwargs):
        if self._q_next_cache is not None:
            return self._q_next_cache
        self._q_next_cache = torch.min(*[
            fn(*args, **kwargs)
            for fn in [self._q_next1, self._q_next2]
        ])
        return self._q_next_cache

    def step(self, *args, **kwargs):
        self._q_next_cache = None
        loss1 = self.net1.step(*args, **kwargs)
        loss2 = self.net2.step(*args, **kwargs)
        return loss1 + loss2


class TD3(DDPG):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, critic=DoubleCritic, **kwargs)
        self.n_step = 0

    def step(self, batch, **kwargs):
        self.n_step += 1

        critic_loss = self.critic.step(batch, actor=self.actor, tau=self.tau)

        actor_loss = 0.0
        if self.n_step == 5:
            self.n_step = 0
            actor_loss = self.actor.step(
                batch, critic=self.critic.net1, tau=self.tau)

        return critic_loss, actor_loss
