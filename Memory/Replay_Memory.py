from collections import deque
import random


class Replay_buffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()

    def append(self, data):
        self.buffer.append(data)

        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()

    def random_sample(self, size):
        mini_batch = random.sample(self.buffer, size)

        return mini_batch