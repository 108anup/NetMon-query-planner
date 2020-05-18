import gurobipy as gp
import random

NUM = 10000
UB = 8

m = gp.Model('test')

xs = m.addVars(range(NUM), vtype=gp.GRB.INTEGER, lb=1, ub=UB, name='x')
ys = m.addVars(range(NUM), vtype=gp.GRB.CONTINUOUS, lb=0, name='y')
zs = [random.random()*50 for i in range(NUM)]

for i in range(NUM):
    m.addConstr(xs[i] * ys[i] == zs[i])

m.setParam(gp.GRB.Param.NonConvex, 2)
ymax = m.addVar(vtype=gp.GRB.CONTINUOUS, name='ymax')
m.addGenConstrMax(ymax, ys)

m.setObjective(ymax)
m.ModelSense = gp.GRB.MINIMIZE

# m.setParam(gp.GRB.Param.MIPFocus, 2)
m.setParam(gp.GRB.Param.TimeLimit, 120)
m.update()
m.optimize()

# """
# Above takes around 30 to 90 seconds on
# AMD Ryzen 5 3600 6-Core Processor (12 H/W Threads)
# 16 GB RAM
# """

# Known optimal solution:
ymax_opt = 0
for i in range(NUM):
    xs[i].lb = UB
    this_ys = zs[i] / UB
    ys[i].start = this_ys
    ymax_opt = max(ymax_opt, this_ys)

ymax.start = ymax_opt
print(ymax_opt)

m.update()
m.optimize()
