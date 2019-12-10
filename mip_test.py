import time
import gurobipy as gp
from gurobipy import GRB


def get_pdt(a, b, pdt_name, m):
    m.update()
    pdt = m.addVar(vtype=GRB.CONTINUOUS,
                   name='pdt_{}'.format(pdt_name))
    loga = m.addVar(vtype=GRB.CONTINUOUS,
                    name='log_{}'.format(a.varName))
    logb = m.addVar(vtype=GRB.CONTINUOUS,
                    name='log_{}'.format(b.varName))
    logpdt = m.addVar(vtype=GRB.CONTINUOUS,
                      name='log_pdt_{}'.format(pdt_name))
    m.addGenConstrLogA(pdt, logpdt, 2,
                       name='log_pdt_{}'.format(pdt_name),
                       options="FuncPieces=-1 FuncPieceError=0.001")
    m.addGenConstrLogA(a, loga, 2,
                       name='log_{}'.format(a.varName))
    m.addGenConstrLogA(b, logb, 2,
                       name='log_{}'.format(b.varName))
    m.addConstr(logpdt == loga + logb,
                name='pdt_{}'.format(pdt_name))
    return pdt


m = gp.Model('pdt_test')
x = m.addVar(vtype=GRB.CONTINUOUS, lb=5, ub=6, name='x')
y = m.addVar(vtype=GRB.CONTINUOUS, lb=6, ub=7, name='y')
z = m.addVar(vtype=GRB.CONTINUOUS, lb=11, ub=11, name='z')
logz = m.addVar(vtype=GRB.CONTINUOUS, name='logz')
ez = m.addVar(vtype=GRB.CONTINUOUS, name='ez')

m.addGenConstrLogA(z, logz, 2)
m.addGenConstrExpA(z, ez, 2)

xy = get_pdt(x, y, 'xy', m)
m.setObjective(x+y+xy, GRB.MINIMIZE)

start = time.time()
m.update()
end = time.time()
print("Model update took: {} seconds".format(end - start))

m.write("test.lp")

start = time.time()
m.optimize()
end = time.time()
print("Model optimize took: {} seconds".format(end - start))

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))
