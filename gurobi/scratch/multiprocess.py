"""# from types import SimpleNamespace
from collections import namedtuple
import concurrent.futures
import multiprocessing as mp
import time


mycls = namedtuple(\\\"mycls\\\", \\\"num\\\")
myarr = [mycls(num=i) for i in range(1000000)]


# class MyClass():
#     def print_len(self):
#         # pass
#         print(len(self.arr))


WORKERS = 4
# objs = [MyClass() for i in range(WORKERS)]
# for obj in objs:
#     obj.arr = myarr


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
    print(len(myarr))
    # print(len(myarr))
    # time.sleep(10)


# PARALLEL PART
st = time.time()
# executer = concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS)
# futures = []
# for i in range(WORKERS):
#     futures.append(executer.submit(myfun, i))
pool = mp.Pool()
pool.map(myfun, [i for i in range(WORKERS)])
print(myarr[:5])
# for i in range(WORKERS):
#     futures[i].result()
parallel = time.time() - st
print(\\\"Parallel: \\\", parallel)

# SEQUENTIAL PART
st = time.time()
list(map(myfun, [i for i in range(WORKERS)]))
print(myarr[:5])
sequential = time.time() - st
print(\\\"Sequential: \\\", sequential)
print(\\\"Overhead: \\\", parallel - sequential)
"""

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


class Test(object):
    class_var = {}


i1 = Test()
i2 = Test()

i1.class_var[4] = 5
print(i2.class_var)
