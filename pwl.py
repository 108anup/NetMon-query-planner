import numpy as np
import re
import pwlf
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import pprint
#import ipdb

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

plt.figure()
plt.plot(X, Y, 'o')
plt.plot(X, finterp(X), '-')
#plt.xscale('log')
#plt.yscale('log')
plt.show(block=True)
