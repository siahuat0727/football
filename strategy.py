import random
import torch

class Strategy:
    def __init__(self, max_eps, min_eps, eps_decay, do_random=True):
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.do_random = do_random
        self.n_step = 0
        self._eps = max_eps

    def step(self):
        self._eps = max(self._eps - (1/self.eps_decay), self.min_eps)
        self.n_step += 1

    @property
    def state(self):
        return 'explore' if self.do_random and self.n_step % 10 != 0 else 'exploit'

    @property
    def eps(self):
        return self._eps if self.state == 'explore' else 0.0

    def choose_act(self, model, s, n_a):
        if random.random() < self.eps:
            a_random = random.randrange(n_a)
            return torch.tensor(a_random).to(s.device), None
        return model.choose_act(s.unsqueeze(0), add_noise=False)
