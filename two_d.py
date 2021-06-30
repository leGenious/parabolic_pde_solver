import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

class MOL:
    def __init__(self, n_space_points):
        self.n_space_points = n_space_points
        self.h = 1/(self.n_space_points+1)
        self.M = lil_matrix( (n_space_points**2, n_space_points**2) )
        for i in range(n_space_points):
            self.M[i,i] = -2
        for i in range(n_space_points-1):
            self.M[i+1,i] = self.M[i,i+1] = 1
        self.M *= 1/(self.h**2)

    def __call__(self, t, y):
        return self.M @ y

def method_of_lines():
    n_space_points = 64

