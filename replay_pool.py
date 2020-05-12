import torch
import random
from collections import deque


class ReplayPool:

    def __init__(self, data_type, capacity=100000, cumulated_reward=False):
        self.capacity = capacity
        self._memory = deque(maxlen=capacity)
        self.capacity = capacity
        self.T = data_type
        self.cumulated_reward = cumulated_reward

        if cumulated_reward:
            self.stack = []
            self.gamma = 0.99 # TODO assign

    def record(self, data, done):
        if not self.cumulated_reward:
            self.append(data)
            return

        self.stack.append(data)
        if not done:
            return

        total_r = None
        while self.stack:
            data = self.stack.pop()  # (s, a, r, s_, done)
            if total_r is None:
                total_r = data[2]
            else:
                total_r = self.gamma*total_r + data[2]
            data[2] = total_r
            self.append(data)

    def append(self, data):
        self._append(self.T(*data))

    def clean(self):
        self._memory = deque(maxlen=self.capacity)

    def _pop(self):
        self._memory.pop()

    def _append(self, data):
        self._memory.append(data)

    def __len__(self):
        return len(self._memory)

    def sample(self, batch_size, cuda=False):
        batch_size = min([len(self), batch_size])
        samples = random.sample(self._memory, batch_size)
        batch = self.T(*_list2tensor(samples, cuda))
        return batch


def _list2tensor(data, cuda):
    data = list(zip(*data))
    data = [torch.stack(d).view(len(d), -1) for d in data]
    if cuda:
        data = [d.cuda() for d in data]
    return data
