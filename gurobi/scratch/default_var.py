from types import SimpleNamespace

b = SimpleNamespace(x=3)


def myfun(x=b.x):
    print(x)


myfun()
myfun(4)
b.x = 7
myfun()
