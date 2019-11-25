'''
r => rows
c => cols
h => hashes
'''

class sketch:
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def __getattr__(self, attr):
        return self.__dict__[attr]


class cm_sketch(sketch):
    def epsilon(self, r, c, h):
        pass
    def delta(self, r, c, h):
        pass
    def rows(self, epsilon, delta):
        pass
    def cols():
        pass

class device:
    def throughput(self, args):
        raise NotImplementedError
    def resources(self, args):
        raise NotImplementedError

class cpu(device):
    def throughput(self, sketches):
        pass
    def resources(self, sketches):
        pass


query = [(cm_sketch,  )]

sketch_requirements = [
    (1, epsilon, delta),
    (2, epsilon, delta)
]

sketches = [
    (1, r, c, h), # Count min like
    (2, [(r1, c1, h1), (r2, c2, h2)]) # Univmon like
]

devices = [
    (1, throughput([(r, c, h)]), resources[(r, c, h)])
]
