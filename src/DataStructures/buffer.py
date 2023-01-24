from typing import List, Iterable


class Buffer:
    buffer: List

    def __init__(self, buffer: Iterable = None):
        if buffer:
            self.buffer = list(buffer)
        else:
            self.buffer = []

    def __bool__(self):
        return bool(self.buffer)

    @property
    def top(self):
        return self.buffer[0]

    def pop(self):
        result = self.top
        self.buffer = self.buffer[1:]
        return result

    def add(self, item):
        self.buffer.append(item)
        return self

    def add_list(self, list_: Iterable):
        for item in list_:
            self.add(item)
        return self

    def __iter__(self):
        return self.buffer
