from src.tools.io import read_file
from src.tools.measurements import UAS, LAS


class Test:
    a = [1,2,3,4]

    """
    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index < len(self.a):
            value = self.a[self.iter_index]
            self.iter_index += 1
            return value
        else:
            raise StopIteration
    """

    def __getitem__(self, item):
        return self.a[item]


def main():
    for element in Test():
        print(element)


if __name__ == '__main__':
    main()

