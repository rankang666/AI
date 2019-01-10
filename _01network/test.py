#coding=utf-8

import numpy as np

y = np.array([1,2])
t = np.array([3,4])
print(y-t)
print(y * t)

d = np.array([0.1,0.1])

f = lambda x: 1.0/x

print((y-t)* f(d))


