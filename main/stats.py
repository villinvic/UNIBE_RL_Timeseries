import numpy as np


class Stats:
    def __init__(self, *stats):
        self.values = {
            name: RunningAvg(name) for name in stats
        }

    def __getitem__(self, item):
        return self.values[item]()

    def __setitem__(self, key, value):
        self.values[key].push(value)

    def reset(self, item):
        self.values[item].reset()


class RunningAvg:
    def __init__(self, name, length=100):
        self.name = name
        self.length = length
        self.values = np.full((length,), fill_value=np.nan, dtype=np.float32)
        self.index = 0

    def push(self, value):
        self.values[self.index % self.length] = value
        self.index += 1

    def __call__(self):
        return np.nanmean(self.values)

    def __repr__(self):
        return "| %s = %.3f |" % (self.name.rjust(8, ' '), self())

    def reset(self):
        self.__init__(self.name, self.length)
