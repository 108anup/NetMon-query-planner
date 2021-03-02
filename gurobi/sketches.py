import math

from common import Namespace, constants


class cm_sketch(Namespace):

    def __repr__(self):
        return 'CM_{}'.format(self.sketch_id)

    def details(self):
        return "cm: eps0: {}, del0: {}, memory_per_row: {}".format(
            self.eps0, self.del0, self.memory_per_row())

    # Used for partitioning only
    def rows(self):
        return math.ceil(math.log(1/self.del0))

    # https://stackoverflow.com/questions/2064202/private-members-in-python
    def __cols(self):
        return math.ceil(math.e / self.eps0)

    # @memoize
    # Used for rounding by powers of 2
    def memory_per_row(self):
        return constants.cell_size * self.__cols() / constants.KB2B

    # Used for static allocation
    def total_mem(self, num_rows=None):
        if(num_rows is None):
            num_rows = self.rows()
        return self.memory_per_row() * num_rows

    # Used for memory access time
    def uniform_mem(self, num_rows=None):
        return self.total_mem(num_rows)

    # Used for perf per packet, and static allocation
    # NOTE: All levels are compressed into 1 row so
    # static hashes is same as hashes_per_packet
    def hashes_per_packet(self, num_rows=None):
        if(num_rows is None):
            num_rows = self.rows()
        return num_rows

    # Used for perf per packet
    def mem_updates_per_packet(self, num_rows=None):
        if(num_rows is None):
            num_rows = self.rows()
        return num_rows


class cs_sketch(cm_sketch):

    def __repr__(self):
        return 'CS_{}'.format(self.sketch_id)

    def details(self):
        return "cs: eps0: {}, del0: {}, memory_per_row: {}".format(
            self.eps0, self.del0, self.memory_per_row())

    def hashes_per_packet(self, num_rows=None):
        if(num_rows is None):
            num_rows = self.rows()
        return 2 * num_rows


class univmon(cs_sketch):

    def __repr__(self):
        return 'Univmon_{}'.format(self.sketch_id)

    def details(self):
        return "univmon: eps0: {}, del0: {}, memory_per_row: {}".format(
            self.eps0, self.del0, self.memory_per_row())

    def memory_per_row(self):
        return super(univmon, self).memory_per_row() * self.levels
        # return constants.cell_size * self.__cols() / constants.KB2B

    def total_mem(self, num_rows=None):
        if(num_rows is None):
            num_rows = self.rows()
        return self.memory_per_row() * num_rows * self.levels

    def uniform_mem(self, num_rows=None):
        return self.total_mem(num_rows) / 2

    def hashes_per_packet(self, num_rows=None):
        if(num_rows is None):
            num_rows = self.rows()
        return 2 * num_rows + 1  # 1 for level identification
