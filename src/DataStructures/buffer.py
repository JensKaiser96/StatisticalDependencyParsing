from typing import List, Iterable


class Buffer:
    data: List

    def __init__(self, buffer: Iterable = None):
        if buffer:
            self.data = list(buffer)
        else:
            self.data = []

    def __bool__(self):
        return bool(self.data)

    @property
    def top(self):
        return self.data[0]

    def pop(self):
        result = self.top
        self.data = self.data[1:]
        return result

    def add(self, item):
        self.data.append(item)
        return self

    def add_list(self, list_: Iterable):
        for item in list_:
            self.add(item)
        return self

    def __iter__(self):
        return self.data
