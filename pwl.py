import numpy as np
import re
import pwlf
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import pprint
import ipdb

f = open("list_traversal.data", 'r')
data = f.readlines()
f.close()

regex = re.compile("([0-9]+)\s+([0-9]+\.[0-9]+)")
X = []
Y = []

for line in data[2:-1]:
    line = line.split('#')[0]
    m = regex.match(line)
    x = m.groups()[0]
    y = m.groups()[1]

    X.append(x)
    Y.append(y)

X = np.array(X, dtype=int)
Y = np.array(Y, dtype=float)
m={}
for i in range(len(X)):
    m[X[i]] = Y[i]

# plt.figure()
# plt.plot(X, Y, '-')
# plt.xscale('log')
# plt.show(block=True)

pprint.pprint(m)

pts = [X.min()/1024, 32, 1482912/1024, 5931648/1024, 33554432/1024, X.max()/1024]
bpts = np.array([1024 * x for x in pts])
vals = np.array([m[x] for x in bpts])
print(pts, vals)

# lgpts = np.log2(bpts)
# lgvals = np.log2(vals)

finterp = interp1d(bpts, vals)

# X = np.log2(X)
# Y = np.log2(Y)

L1_size = 32 * 1024
L2_size = 256 * 1024
L3_size = 8192 * 1024

L1_ns = 0.510892
L2_ns = 1.6
L3_ns = 5.84114
L4_ns = 38.9213


def predicted(X):
    Y = np.array(X, dtype=float)
    for (i, M) in enumerate(X):
        r1 = (min(L1_size, M)) / M
        r2 = (max(0, min(L2_size - L1_size, M - L1_size)) / M)
        r3 = (max(0, min(L3_size - L2_size, M - L2_size)) / M)
        r4 = (max(0, M - L3_size) / M)
        Y[i] = r1 * L1_ns + r2 * L2_ns + r3 * L3_ns + r4 * L4_ns
    return Y


Y_pdt = predicted(X)

plt.figure()
plt.plot(X, Y, 'o', label='Ground Truth')
plt.plot(X, Y_pdt, '-', label='Uniform access model')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Array Size (Bytes)')
plt.ylabel('Time to access a random element of array (ns)')
plt.legend()
#plt.show(block=True)

#plt.savefig('mem-access.pdf')

ipdb.set_trace()
np.sum(np.abs((Y_pdt - Y) / Y))
