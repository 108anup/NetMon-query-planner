import math

from common import constants, memoize, Namespace


class cm_sketch(Namespace):

    def __repr__(self):
        return 'CM_{}'.format(self.sketch_id)

    def details(self):
        return "cm: eps0: {}, del0: {}, min_mem: {}".format(
            self.eps0, self.del0, self.min_mem())

    def rows(self):
        return math.ceil(math.log(1/self.del0))

    def cols(self):
        return math.ceil(math.e / self.eps0)

    # @memoize
    def min_mem(self):
        return constants.cell_size * self.cols() / constants.KB2B


class cs_sketch(Namespace):

    def __repr__(self):
        return 'CS_{}'.format(self.sketch_id)

    def details(self):
        return "cm: eps0: {}, del0: {}, min_mem: {}".format(
            self.eps0, self.del0, self.min_mem())

    def rows(self):
        return math.ceil(math.log(1/self.del0))

    def cols(self):
        return math.ceil(3 / (self.eps0**2))

    # @memoize
    def min_mem(self):
        return constants.cell_size * self.cols() / constants.KB2B
