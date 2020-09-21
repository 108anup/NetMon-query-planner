def freeze_object(obj):
    # import ipdb; ipdb.set_trace()
    if(isinstance(obj, Namespace)):
        obj.__setattr__ = obj.frozen_setter
    else:
        raise TypeError("obj: {} is not a Namespace instance", obj)


class Namespace(object):
    __isfrozen = False

    def __init__(self, **kw):
        self.__dict__.update(kw)

    # def __getattr__(self, attr):
    #     if attr in self.__dict__:
    #         return self.__dict__[attr]
    #     else:
    #         raise AttributeError("Not found key: {}".format(attr))

    # def frozen_setter(self, attr):
    #     raise AttributeError("Object {} is frozen,"
    #                          " can't set attribute:"
    #                          .format(self, attr))

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen object" % self)
        object.__setattr__(self, key, value)

    def __repr__(self):
        return repr(self.__dict__)

    def _freeze(self):
        self.__isfrozen = True


myobj = Namespace(x=3)
print(myobj)

myobj._freeze()

myobj.y = 5
print(myobj)
