import time
import gurobipy as gp
from gurobipy import GRB


def get_pdt(a, b, pdt_name, m):
    m.update()
    pdt = m.addVar(vtype=GRB.CONTINUOUS,
                   name='pdt_{}'.format(pdt_name))
    # loga = m.addVar(vtype=GRB.CONTINUOUS,
    #                 name='log_{}'.format(a.varName), lb=-GRB.INFINITY)
    # logb = m.addVar(vtype=GRB.CONTINUOUS,
    #                 name='log_{}'.format(b.varName), lb=-GRB.INFINITY)
    # logpdt = m.addVar(vtype=GRB.CONTINUOUS,
    #                   name='log_pdt_{}'.format(pdt_name), lb=-GRB.INFINITY)
    # m.addGenConstrExpA(logpdt, pdt, 2,
    #                    name='log_pdt_{}'.format(pdt_name),
    #                    options="FuncPieces=-1 FuncPieceError=0.01")
    # m.addGenConstrExpA(loga, a, 2,
    #                    name='log_{}'.format(a.varName),
    #                    options="FuncPieces=-1 FuncPieceError=0.01")
    # m.addGenConstrExpA(logb, b, 2,
    #                    name='log_{}'.format(b.varName),
    #                    options="FuncPieces=-1 FuncPieceError=0.01")
    # m.addConstr(logpdt == loga + logb,
    #             name='pdt_{}'.format(pdt_name))
    m.addQConstr(pdt == a * b,
                 name='pdt_{}'.format(pdt_name))

    return pdt


m = gp.Model('pdt_test')
m.setParam(GRB.Param.NonConvex, 2)
x = m.addVar(vtype=GRB.CONTINUOUS, name='x', lb=300)
y = m.addVar(vtype=GRB.CONTINUOUS, name='y', lb=4.32)
z = m.addVar(vtype=GRB.CONTINUOUS, name='z')
logz = m.addVar(vtype=GRB.CONTINUOUS, name='logz', lb=-1, ub=-0.5)
ez = m.addVar(vtype=GRB.CONTINUOUS, name='ez')

m.addGenConstrLogA(z, logz, 2)
m.addGenConstrExpA(z, ez, 2)

xy = get_pdt(x, y, 'xy', m)
m.setObjective(x + y + xy, GRB.MINIMIZE)

start = time.time()
m.update()
end = time.time()
print("Model update took: {} seconds".format(end - start))

m.Params.Presolve = 0

m.write("test.lp")

start = time.time()
m.optimize()
end = time.time()
print("Model optimize took: {} seconds".format(end - start))

for v in m.getVars():
    print('{} {}'.format(v.varName, v.x))

print("Obj: {}".format(m.getObjective().getValue()))

m.printQuality()
