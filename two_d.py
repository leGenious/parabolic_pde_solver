import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp

"""
This module contains solvers for heat equation problems in two dimensions with
homogeneous Dirichlet boundary conditions.
It is rather rudimentary in nature, in that it only deals with a very narrow
band of problems.

Author
------
Daniel Walsken

Date
----
01.07.2021

Version
-------
Pretty much untested apart from a few edge cases.
"""

class MOL:
    """
    This class is a wrapper for a function with static data. In order to
    limit the matrix construction to once per program run, this seemed like the
    most reasonable approach.
    """
    def __init__(self, n_space_points):
        """
        The init handles most of the workload for the computation, in that it
        constructs the (constant) matrix in an efficient way.

        Parameters
        ----------
        n_space_points : int
            The number of internal grid points per dimension. The grid points
            are equally spaced, such that h = 1/(n_space_points)
        """
        self.n_space_points = n_space_points
        self.n_squared = self.n_space_points**2
        self.h = 2/(self.n_space_points+1)
        self.M = lil_matrix( (self.n_squared, self.n_squared) )
        for i in range(self.n_squared):
            # diagonal
            self.M[i,i] = -4
        for i in range(self.n_squared-1):
            # first off-diagonal
            self.M[i+1,i] = self.M[i,i+1] = 1
        for i in range(self.n_squared - self.n_space_points):
            # off diagonal for the top-bot relations
            self.M[i+self.n_space_points,i] = 1
            self.M[i,i+self.n_space_points] = 1
        self.M = csc_matrix(self.M)
        self.M /= self.h**2

    def __call__(self, t, y):
        """
        This method is supposed to be passed to the ode_solver

        Parameters
        ----------

        t : float
            time t, not required for the calculation due to homogeneous
            constant boundaries, but required for the ode solver
        y : array_like, shape(n_space_points,)
            the previous value

        Returns
        -------

        out : array_like, shape(n_space_points,)
            the laplacian applied to the discretized field.
        """
        return self.M @ y

class Rothe:
    """
    Solver class for Rothes method FDM.
    """
    def __init__(self, n_space_points, n_time_points, u_initial=None):
        """
        Initialize the Rothe FDM solver

        Parameters
        ----------
        n_space_points : int
            The number of space points in each of the two dimensions
        n_time_points : int
            The number of time points between 0 and 1
        u_initial : array_like, shape(n_space_points*n_space_points) or None
            The initial data for the calculation
        """
        self.nsp = n_space_points
        self.nsq = self.nsp**2
        self.ntp = n_time_points
        self.h = 1/(self.nsp + 1)
        self.k = 1/self.ntp
        self.r = self.k/(self.h**2)
        self.M = lil_matrix( (self.nsq, self.nsq) )
        for i in range(self.nsq):
            self.M[i,i] = 4*self.r+1
        for i in range(self.nsq-1):
            self.M[i+1,i] = self.M[i,i+1] = -self.r
        for i in range(self.nsq - self.nsp):
            self.M[i+self.nsp,i] = self.M[i,i+self.nsp] = -self.r
        self.M = csc_matrix(self.M)
        self.u_initial = u_initial

    def solve(self, u_initial=None):
        """
        Solve the system arising from Rothes Method FDM with homogeneous
        Dirichlet boundary conditions

        Parameters
        ----------
        u_initial : array_like, shape(self.nsq) or None
            The initial data, if not already manually supplied to the solver
            class during initialization. If initial data has already been
            assigned and new initial data is passed with this function, the old
            initial data will be overwritten and the calculation is carried out

        Returns
        -------
        out : array_like, shape(self.nsq, self.ntp)
            The resulting temporal snapshots of solving the heat equation.
        """
        if u_initial is None and self.u_initial is None:
            raise Exception("Please initialize the data before solving .__.")
        self.u = np.zeros( (self.nsq, self.ntp) )
        if self.u_initial is None:
            self.u_initial = u_initial
            self.u[:,0] = self.u_initial
        if u_initial is None:
            self.u[:,0] = self.u_initial
        for i in range(1,self.ntp):
            self.u[:,i] = spsolve(self.M, self.u[:,i-1])
        return self.u

def init_y0(n_space_points):
    """
    Method init_y0. This method initializes a wave packet in the center of the
    grid.

    Parameters
    ----------
    n_space_points : int
        The number of space points in every of the two dimensions

    Returns
    -------
    out : np.array, shape(n_space_points*n_space_points)
        The resulting initialized array in lexographic ordering
    """
    y0 = np.zeros( (n_space_points**2) )
    h = 2/(n_space_points+1)
    for i in range(n_space_points):
        y = h + i * h - 1
        for j in range(n_space_points):
            ind = i * n_space_points + j
            x = h + j * h - 1
            r = (x**2 + y**2)**0.5
            if r < 0.25:
                y0[ind] = 10*np.exp(1 / (r**2 - 0.0625))
            else:
                y0[ind] = 0
    return y0

def plot(n_space_points, y):
    """
    Rudimentary plotting routine for testing purposes.

    Parameters
    ----------

    n_space_points : int
        The number of internal space points in each of the two dimensions
    y : array_like, shape(n_space_points*n_space_points,)
        The array to be plotted in lexographic ordering.
    """
    y = y.reshape(n_space_points, n_space_points)
    plt.imshow(y)
    plt.show()

def test_method_of_lines():
    """
    Simple, no parameter testing routine. The expected output is that a wave
    dissipates.
    This stuff works, don't try it with 'RK45' or similar explicit method, it
    won't work (since they cannot deal with the arising stiff system properly).
    """
    n_space_points = 64
    ode = MOL(n_space_points)
    plt.spy(ode.M, markersize=2)
    plt.show()
    y0 = init_y0(n_space_points)
    plot(n_space_points,y0)
    result = solve_ivp(ode, (0,1), y0, method='LSODA')
    for i in range( 0, len(result.t), len(result.t)//10 ):
        plot(n_space_points, result.y[:,i])

def test_rothe_method():
    """
    Simple, no parameter testing routine. The expected output is that a wave
    dissipates.
    """
    n_space_points = 64
    n_time_points = 100
    u_initial = init_y0(n_space_points)
    solver = Rothe(n_space_points, n_time_points)
    u = solver.solve(u_initial)
    for i in range(10):
        plot(n_space_points, u[:,i])

def main():
    test_method_of_lines()
    test_rothe_method()

if __name__ == "__main__":
    main()
