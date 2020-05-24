import torch
import random
from collections import deque, namedtuple


class ReplayPool:

    # namedtuple('Data', ['s', 'a', 'r', 's_', 'logprobs'])
    def __init__(self, capacity=100000, cumulated_reward=False):
        self.capacity = capacity
        self._memory = deque(maxlen=capacity)
        self.capacity = capacity
        self.cumulated_reward = cumulated_reward
        self._type = []

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

        prev_r = 0.0
        while self.stack:
            data = self.stack.pop()
            data['r'] += self.gamma * prev_r
            prev_r = data['r']
            self.append(data)

    def append(self, data):
        clean_data = [
            data[key]
            for key in self._type
        ]
        self._append(self.convert(clean_data))

    def register(self, *it):
        self._type.extend(it)

    def convert(self, data):
        return namedtuple('Data', self._type)(*data)

    def _pop(self):
        self._memory.pop()

    def _append(self, data):
        self._memory.append(data)

    def sample(self, batch_size):
        batch_size = min([len(self), batch_size])
        samples = random.sample(self._memory, batch_size)
        batch = self.convert(_list2tensor(samples))
        return batch

    def clean(self):
        self._memory = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self._memory)



def _list2tensor(data):
    data = list(zip(*data))
    data = [torch.stack(d).view(len(d), -1) for d in data]
    return data
