import copy
import random
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

    def __init__(self, in_dim, hid_dims, out_dim, activation=nn.ReLU(), **kwargs):  # TODO
        super().__init__()
        self.layers = _make_layers([in_dim, *hid_dims, out_dim], activation)

    def forward(self, x):
        return self.layers(x)


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
        return v + (a - a.mean(dim=1, keepdim=True).detach())


class _Net(nn.Module, ABC):
    def __init__(self, net, lr=1e-3, **kwargs):
        super().__init__()
        self.net = net
        self.optim = Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)

    @abstractmethod
    def _loss(self, batch, **kwargs):
        pass

    def step(self, batch, **kwargs):
        loss = self._loss(batch, **kwargs)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optim.step()
        return loss.item()


class _QNet(_Net):
    def __init__(self, *args, update_freq=50, **kwargs):  #TODO update
        super().__init__(*args, **kwargs)

        self._loss_fn = nn.MSELoss()
        self.update_freq = update_freq

        self.n_step = 0
        self.net_t = None
        self.update_net_t()

    def update_net_t(self):
        self.net_t = copy.deepcopy(self.net).eval()
        for param in self.net_t.parameters():
            param.requires_grad = False

    def choose_act(self, s, **kwargs):
        return self.net_t(s).squeeze().argmax(), None

    def _loss(self, batch, gamma=0.999):
        q_eval = self._q_eval(batch.s, batch.a)
        q_target = self._q_target(batch.r, batch.s_, gamma)
        return self._loss_fn(q_eval, q_target)

    def step(self, *args, **kwargs):
        self.n_step += 1

        ret = super().step(*args, **kwargs)

        if self.n_step % self.update_freq == 0:
            self.update_net_t()
            self.n_step = 0

        return ret


class _QLearningMixin:
    def _q_eval(self, s, a):
        return self.net(s).gather(1, a)

    def _q_next(self, s_):
        return self.net_t(s_).detach().max(dim=1, keepdim=True)[0]

    def _q_target(self, r, s_, gamma):
        return r + gamma*self._q_next(s_)


class _DQLearningMixin(_QLearningMixin):
    def _q_next(self, s_):
        a = self.net(s_).argmax(dim=1, keepdim=True)
        return self.net_t(s_).gather(1, a)


class DQN(_QLearningMixin, _QNet):
    def __init__(self, *args, **kwargs):
        _net = _MLPNet(*args, **kwargs)
        super().__init__(_net, *args, **kwargs)


class DDQN(_DQLearningMixin, _QNet):
    def __init__(self, *args, **kwargs):
        _net = _MLPNet(*args, **kwargs)
        super().__init__(_net, *args, **kwargs)


class DuelingDQN(_QLearningMixin, _QNet):
    def __init__(self, *args, **kwargs):
        _net = _DuelingNet(*args, **kwargs)
        super().__init__(_net, *args, **kwargs)


class DuelingDDQN(_DQLearningMixin, _QNet):
    def __init__(self, *args, **kwargs):
        _net = _DuelingNet(*args, **kwargs)
        super().__init__(_net, *args, **kwargs)


class ActorCritic(nn.Module):
    def __init__(self, in_dim, hid_dims, out_dim, **kwargs):
        super().__init__()
        self.actor = _MLPNet(in_dim, hid_dims, out_dim, activation=nn.Tanh())
        self.critic = _MLPNet(in_dim, hid_dims, 1, activation=nn.Tanh())

    def _s2d(self, s, a=None):
        return Categorical(nn.Softmax(dim=-1)(self.actor(s)))

    def _a2p(self, a, dist):
        a_ = a.squeeze()
        assert a_.dim() <= 1, f'size = {a.size()}'
        return dist.log_prob(a_).view(a.size())

    def choose_act(self, s):
        with torch.no_grad():
            dist = self._s2d(s)
        a = dist.sample()

        logprobs = self._a2p(a, dist)
        # return a, logprobs
        return a, logprobs #TODO

    def analysis(self, s, a):
        dist = self._s2d(s)
        logprobs = self._a2p(a, dist)
        return logprobs, dist.entropy().view(a.size())

    def s2v(self, s):
        return self.critic(s)


class PPO(_Net):
    def __init__(self, **kwargs):
        net = ActorCritic(**kwargs)
        super().__init__(net, **kwargs)

        self.mse_loss = nn.MSELoss(reduction='none')
        self.clip_eps = 0.2

    def _loss(self, batch, **kwargs):

        logprobs, entropy = self.net.analysis(batch.s, batch.a)
        v = self.net.s2v(batch.s)

        assert logprobs.size() == batch.logprobs.size(), f'{logprobs.size()}, {batch.logprobs.size()}'
        ratios = torch.exp(logprobs - batch.logprobs)

        # advantage A(s,a) = R + yV(s') - V(s)
        # r = (batch.r - batch.r.mean()) / batch.r.std()  # TODO normalized?
        advs = batch.r - v.detach()
        assert batch.r.size() == v.size() == ratios.size(), \
            f'{v.size()} {batch.r.size()} {ratios.size()}'

        actor_loss1 = ratios * advs
        actor_loss2 = torch.clamp(ratios, 1-self.clip_eps, 1+self.clip_eps) * advs

        actor_loss = torch.min(actor_loss1, actor_loss2)  # TODO better name?
        critic_loss = 0.5 * self.mse_loss(v, batch.r)
        entropy_loss = 0.001 * entropy

        # print()
        # print('---')
        # print('a', actor_loss.mean().item())
        # print('c', critic_loss.mean().item())
        # print('e', entropy_loss.mean().item())

        loss = critic_loss - actor_loss - entropy_loss
        return loss.mean()

    def choose_act(self, *args, **kwargs):
        return self.net.choose_act(*args, **kwargs)


class Actor(_Net):
    def __init__(self, *args, **kwargs):
        _net = _MLPNet(*args, activation=nn.Tanh(), **kwargs)
        super().__init__(_net, **kwargs)

    def forward(self, *args, add_noise=True, **kwargs):
        x = self.net(*args, **kwargs)
        if add_noise:
            x += torch.randn(x.size()).to(x.device) * 0.1  # TODO cuda
        x = torch.nn.functional.gumbel_softmax(x, hard=True)
        return x

    def _loss(self):
        pass


class Critic(_Net):
    def __init__(self, **kwargs):
        in_dim = kwargs.pop('in_dim') + kwargs.pop('out_dim')
        _net = _MLPNet(in_dim=in_dim, out_dim=1, activation=nn.Tanh(), **kwargs)
        super().__init__(_net, **kwargs)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))

    def _loss(self):
        pass


class DDPG(nn.Module):

    def __init__(self, *args, tau=0.01, **kwargs):
        super().__init__()

        self.n_a = kwargs.get('out_dim')
        self.actor = Actor(*args, **kwargs)  #TODO lra, lrc
        self.critic = Critic(*args, **kwargs)

        self.actor_t = copy.deepcopy(self.actor).eval()
        self.critic_t = copy.deepcopy(self.critic).eval()
        self.tau = tau
        self._loss_fn = nn.MSELoss()  #TODO duplicate

    def choose_act(self, s, add_noise=True):
        return self.actor_t(s, add_noise=add_noise).argmax(), None

    def _loss_critic(self, batch, gamma=0.99, **kwargs):

        q_eval = self.critic(batch.s, _onehot(batch.a, self.n_a))  # same as dqlearning  # n_a
        q_next = self.critic_t(batch.s_, self.actor_t(batch.s_))
        assert q_eval.size() == q_next.size(), f'{q_eval.size()}, {q_next.size()}'
        assert q_eval.size() == batch.r.size(), f'{q_eval.size()}, {batch.r.size()}'
        q_target = batch.r + gamma * q_next.detach()
        return self._loss_fn(q_eval, q_target)

    def _loss_actor(self, batch, **kwargs):
        act_logits = self.actor(batch.s)

        loss = -self.critic(batch.s, act_logits).mean() #TODO regularize?+ 1e-3*(act_logits**2).mean()  #TODO l2norm
        return loss

    def _loss(self):
        print('Please implement')
        pass

    def soft_update(self):
        def do_soft_update(src_net, dst_net, tau):
            for src, dst in zip(src_net.parameters(), dst_net.parameters()):
                dst.data.copy_(dst.data * (1.0 - tau) + src.data * tau)
        do_soft_update(self.actor, self.actor_t, self.tau)
        do_soft_update(self.critic, self.critic_t, self.tau)


    def step(self, batch, **kwargs):
        critic_loss = self._loss_critic(batch, **kwargs)
        self.critic.optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic.optim.step()

        actor_loss = self._loss_actor(batch)
        self.actor.optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor.optim.step()

        self.soft_update()
        return critic_loss.item(), actor_loss.item()


class Critic2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net1 = Critic(*args, **kwargs)
        self.net2 = Critic(*args, **kwargs)

    def forward(self, s, a):
        return self.net1(s, a), self.net2(s, a)


class TD3(DDPG):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic = Critic2(*args, **kwargs)
        self.critic_t = copy.deepcopy(self.critic).eval()
        self.n_step = 0

    def _loss_critic(self, batch, gamma=0.99, **kwargs):

        q_eval1, q_eval2 = self.critic(batch.s, _onehot(batch.a, self.n_a))
        q_next1, q_next2 = self.critic_t(batch.s_, self.actor_t(batch.s_))

        q_target = batch.r + gamma * torch.min(q_next1, q_next2).detach()
        return self._loss_fn(q_eval1, q_target) + self._loss_fn(q_eval2, q_target)

    def _loss_actor(self, batch, **kwargs):
        act_logits = self.actor(batch.s)

        loss = -self.critic.net1(batch.s, act_logits).mean() #TODO regularize?+ 1e-3*(act_logits**2).mean()  #TODO l2norm
        return loss

    def step(self, batch, **kwargs):
        self.n_step += 1
        critic_loss = self._loss_critic(batch, **kwargs)
        self.critic.net1.optim.zero_grad()
        self.critic.net2.optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic.net1.optim.step()
        self.critic.net2.optim.step()

        if self.n_step % 5 == 0:  #TODO
            actor_loss = self._loss_actor(batch)
            self.actor.optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor.optim.step()
        else:
            actor_loss = torch.tensor(0)

        self.soft_update()
        return critic_loss.item(), actor_loss.item()
