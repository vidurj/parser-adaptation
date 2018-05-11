import collections
import parse

class Vocabulary(object):
    def __init__(self):
        self.frozen = False
        self.values = []
        self.indices = {}
        self.counts = collections.defaultdict(int)

    @property
    def size(self):
        return len(self.values)

    def value(self, index):
        assert 0 <= index < len(self.values)
        return self.values[index]

    def index(self, value):
        if value == 'NUM' or value == "#":
            value = 'num'

        if value == '-LRB-' or value == '-RRB-':
            value = value[1:-1]


        if not self.frozen:
            self.counts[value] += 1

        if value in self.indices:
            return self.indices[value]

        elif not self.frozen:
            self.values.append(value)
            self.indices[value] = len(self.values) - 1
            return self.indices[value]

        elif '|' in value:
            return self.index(value.split('|')[0])

        else:
            print("Unknown value: {}".format(value))
            print(self.counts)
            assert isinstance(value, tuple)
            assert len(value) > 1
            return self.index(value[1:])

    def count(self, value):
        return self.counts[value]

    def freeze(self):
        self.frozen = True
