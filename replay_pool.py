import torch
import random
from collections import deque


class ReplayPool:

    def __init__(self, data_type, capacity=100000, cumulated_reward=False):
        self._memory = deque(maxlen=capacity)
        self.capacity = capacity
        self.T = data_type
        self.stack = []
        self.cumulated_reward = cumulated_reward
        self.gamma = 0.8 # TODO assign

    def record(self, data, done):
        if not self.cumulated_reward:
            self.append(data)
            return

        if not done:
            self.stack.append(data)
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

    def _pop(self):
        self._memory.pop(0)

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
