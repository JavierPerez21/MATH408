import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def norm(vector):
    return sp.simplify(sp.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2))

def normalize(vector):
    vector = [sp.simplify(v) for v in vector]
    vector_norm = norm(vector)
    vector = [sp.simplify(v/vector_norm) for v in vector]
    return np.array(vector)

def diff(vector, wrt='t'):
    return np.array([sp.simplify(sp.diff(vector[0], wrt)),
                     sp.simplify(sp.diff(vector[1], wrt)),
                     sp.simplify(sp.diff(vector[2], wrt))])
