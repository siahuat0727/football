import random

import torch

class ReplayPool:
    def __init__(self, capacity=100000):
        self._memory = []
        self.capacity = capacity

    def append(self, data):
        if len(self) == self.capacity:
            self._pop()
        self._append(data)

    def _pop(self):
        self._memory.pop(0)

    def _append(self, data):
        self._memory.append(data)

    def __len__(self):
        return len(self._memory)

    def sample(self, batch_size, cuda=False):
        samples = random.sample(self._memory, batch_size)
        batch = _list2tensor(samples, cuda)
        return batch

def _list2tensor(data, cuda):
    data = list(zip(*data))
    data = [torch.stack(d).view(len(d), -1) for d in data]
    if cuda:
        data = [d.cuda() for d in data]
    return data
