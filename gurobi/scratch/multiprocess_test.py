# from types import SimpleNamespace
from collections import namedtuple
import concurrent.futures
import multiprocessing as mp
import time
from pathos.pools import ProcessPool
from gurobipy import GRB, tupledict, tuplelist


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


mycls = namedtuple("mycls", "num")
myarr = [Namespace(num=-1) for i in range(1000000)]
myarr_n = Namespace(myar=myarr)

# class MyClass():
#     def print_len(self):
#         # pass
#         print(len(self.arr))


WORKERS = 4
# objs = [MyClass() for i in range(WORKERS)]
# for obj in objs:
#     obj.arr = myarr


_probs = []
def myfun(i):
    # print(i, id(myarr), id(myarr[i]))
    # myarr[i].num = 100
    # print(i, id(myarr), id(myarr[i]))
    # myobj = MyClass()
    # myobj.arr = myarr
    # objs[i].arr[i] = 100
    # print(objs[i].arr[i])
    # objs[i].print_len()
    # objs[i].arr[i].hello = 3
    print(len(_probs[i].myar))
    # print(len(myarr))
    # time.sleep(10)
    # time.sleep(0.8)
    # myn = Namespace(num=i, t=tupledict())
    # myn.t[1,2,3] = i
    return 1


for i in range(WORKERS):
    _probs.append(myarr_n)

# PARALLEL PART
st = time.time()
# executer = concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS)
# futures = []
# for i in range(WORKERS):
#     futures.append(executer.submit(myfun, i))
pool = ProcessPool()
res = pool.map(myfun, [i for i in range(WORKERS)])
print(myarr[:5])
# for i in range(WORKERS):
#     futures[i].result()
parallel = time.time() - st
print("Parallel: ", parallel)

# SEQUENTIAL PART
st = time.time()
print(list(map(myfun, [i for i in range(WORKERS)])))
print(myarr[:5])
sequential = time.time() - st
print("Sequential: ", sequential)
# print("Overhead: ", parallel - sequential)


# from scipy.interpolate import interp1d
# import numpy as np
# import matplotlib.pyplot as plt


# def cache(func):
#     func.my_cache = {}
#     def ret_func(x):
#         if(x in func.my_cache):
#             return func.my_cache[x]
#         else:
#             func.my_cache[x] = func(x)
#             return func.my_cache[x]
#     return ret_func


# xs = list(range(5))
# ys = list(map(lambda x: x*x, xs))
# f = cache(interp1d(xs, ys))
# x = np.arange(0, 4, 0.1)
# # print(f(x))
# y = list(map(f, x))

# plt.plot(xs, ys, 'ro')
# plt.plot(x, y, '-')
# plt.show()


# class Test(object):
#     class_var = {}


# i1 = Test()
# i2 = Test()

# i1.class_var[4] = 5
# print(i2.class_var)
