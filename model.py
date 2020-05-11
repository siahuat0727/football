import copy
import random
from abc import ABC, abstractmethod
from gumbel import * #TODO

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.distributions import Categorical


def _make_layers(dims, activation):
    layers = []
    for idx, (dim_in, dim_out) in enumerate(zip(dims, dims[1:])):
        if idx != 0:
            layers.append(activation)
        layers.append(nn.Linear(dim_in, dim_out))
    return nn.Sequential(*layers)


class _FCNet(nn.Module):

    def __init__(self, dims=[], activation=nn.ReLU, **kwargs):  # TODO
        super().__init__()
        self.layers = _make_layers(dims, activation())

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
        return v + (a - a.mean(dim=1, keepdim=True))


class _Net(ABC):

    @abstractmethod
    def _loss(self, batch, **kwargs):
        pass

    def step(self, batch, **kwargs):
        loss = self._loss(batch, **kwargs)

        self.optim.zero_grad()  # TODO init?
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optim.step()
        return loss.item()


class _QNet(nn.Module, _Net):
    def __init__(self, net):
        super().__init__()

        self._loss_fn = nn.MSELoss()

        self.net = net
        self.net_t = None
        self.update_net_t()

    def update_net_t(self):
        self.net_t = copy.deepcopy(self.net).eval()
        for param in self.net_t.parameters():
            param.requires_grad = False

    def eps_greedy(self, s, n_a, eps=0.1, cuda=False):  # TODO eps case in main function
        if random.random() < eps:
            return random.randrange(n_a)
        else:
            if cuda:
                s = s.cuda()
            s = s.unsqueeze(0)
            return self.net_t(s).squeeze().argmax().item()

    def _loss(self, batch, gamma=0.999):
        q_eval = self._q_eval(batch.s, batch.a)
        q_target = self._q_target(batch.r, batch.s_, gamma)
        return self._loss_fn(q_eval, q_target)


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
        net = _FCNet(*args, **kwargs)
        super().__init__(net)


class DDQN(_DQLearningMixin, _QNet):
    def __init__(self, *args, **kwargs):
        net = _FCNet(*args, **kwargs)
        super().__init__(net)


class DuelingDQN(_QLearningMixin, _QNet):
    def __init__(self, *args, **kwargs):
        net = _DuelingFCNet(*args, **kwargs)
        super().__init__(net)


class DuelingDDQN(_DQLearningMixin, _QNet):
    def __init__(self, *args, **kwargs):
        net = _DuelingFCNet(*args, **kwargs)
        super().__init__(net)


class ActorCritic(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.actor = _FCNet(dims, activation=nn.Tanh)
        self.critic = _FCNet(dims[:-1] + [1], activation=nn.Tanh)

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
        return a.item(), logprobs #TODO

    def analysis(self, s, a):
        dist = self._s2d(s)
        logprobs = self._a2p(a, dist)
        return logprobs, dist.entropy().view(a.size())

    def s2v(self, s):
        return self.critic(s)


class PPO(nn.Module, _Net):
    def __init__(self, *args, lr=1e-3, **kwargs):
        super().__init__()
        self.net = ActorCritic(*args, **kwargs)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.clip_eps = 0.2
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.999))

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

        # minus sign for gradient acsent
        loss = critic_loss - actor_loss - entropy_loss

        return loss.mean()

    def choose_act(self, *args, **kwargs):
        return self.net.choose_act(*args, **kwargs)


class Actor(_FCNet):
    def __init__(self, lr=1e-3, **kwargs):
        super().__init__(activation=nn.Tanh, **kwargs)
        self.optim = Adam(self.parameters(), lr=lr)
        self.tanh = nn.Tanh()

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        return self.tanh(x)


class Critic(_FCNet):
    def __init__(self, lr=1e-3, **kwargs):
        dims = kwargs.pop('dims')
        dims[0] = dims[0] + dims[-1]
        dims[-1] = 1
        super().__init__(dims=dims, **kwargs)
        self.optim = Adam(self.parameters(), lr=lr)

    def forward(self, s, a):
        return super().forward(torch.cat([s, a], dim=1))


class DDPG(_Net):

    def __init__(self, lr_a=1e-3, lr_c=1e-3, tau=0.01, **kwargs):
        lr = kwargs.pop('lr') # TODO del
        self.actor = Actor(lr=lr, **kwargs)  #TODO lra, lrc
        self.critic = Critic(lr=lr, **kwargs)

        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.actor_t = copy.deepcopy(self.actor).eval()
        self.critic_t = copy.deepcopy(self.critic).eval()
        self.tau = tau
        self._loss_fn = nn.MSELoss()  #TODO duplicate

    def choose_act(self, s):
        s = s.unsqueeze(0)
        return self.actor_t(s).squeeze().argmax().item(), None

    def _loss_critic(self, batch, gamma=0.99, **kwargs):
        q_eval = self.critic(batch.s, onehot(batch.a, 2))  # same as dqlearning  # n_a
        q_next = self.critic_t(batch.s_, self.actor_t(batch.s_))
        q_target = batch.r + gamma * q_next
        return self._loss_fn(q_eval, q_target)

    def _loss_actor(self, batch, **kwargs):
        act_logits = self.actor(batch.s)
        out = gumbel_softmax(act_logits, hard=True)

        loss = -self.critic(batch.s, out).mean() + 1e-3*(act_logits**2).mean()  #TODO l2norm
        return loss

    def _loss(self):
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
        self.critic.optim.step()

        actor_loss = self._loss_actor(batch)
        self.actor.optim.zero_grad()
        actor_loss.backward()
        self.actor.optim.step()

        self.soft_update()
        # return critic_loss.item(), actor_loss.item() #TODO
        return critic_loss.item()
