import numpy as np
from matplotlib import pyplot as plt
from pysph.solver.utils import load


def radius(x, y):
    return x*x + y*y


data = load('./case2_method2_output/case2_method2_22163.npz')

pa = data['arrays']['fluid']

m = pa.m
x = pa.x
y = pa.y
N = pa.N
p = pa.p
n = len(m)

count_in = 0
count_out = 0
p_in = 0
p_out = 0

for i in range(n):
    r = radius(x[i], y[i])
    if N[i] < 1:
        if radius(x[i], y[i]) < 0.0625:
            p_in += p[i]
            count_in += 1
        else:
            p_out += p[i]
            count_out += 1
    else:
        continue
print p_in/count_in
print p_out/count_out

print (p_in/count_in) - (p_out/count_out)
