from pathos.pools import ProcessPool

class Namespace:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    # def __getattr__(self, attr):
    #     if attr in self.__dict__:
    #         return self.__dict__[attr]
    #     else:
    #         raise AttributeError("Not found key: {}".format(attr))

    def frozen_setter(self, attr):
        raise AttributeError("Object {} is frozen,"
                             " can't set attribute:"
                             .format(self, attr))

    def __repr__(self):
        return repr(self.__dict__)

WORKERS = 8

def runner(i):
    pass

pool = ProcessPool(WORKERS)
pool.map(runner, [Namespace(i=i) for i in range(WORKERS)])
