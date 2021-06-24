import numpy as np
import matplotlib.pyplot as plt
import logging
from enum import Enum
from matplotlib.animation import FuncAnimation
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import inv
from typing import Callable


"""
One dimensional Parabolic solver for the numerical PDE lecture 2021
By Daniel Walsken
    This program solves the heat equation on the (0,1)x[0,1] domain
    Ix[t_0, t_end]. For reasons of efficiency the methods euler_forward,
    euler_backward and crank_nicolson are implemented not as one general theta
    method but as individual methods.
"""

class Solver_type(Enum):
    # Crank Nicolson
    CN = "Crank Nicolson"
    # Implicit Euler (Euler Backwards)
    EB = "Euler Backward"
    # Explicit Euler (Euler Forwards)
    EF = "Euler Forward"

def get_r(n_space_points,n_time_points):
    h = 1/(1+n_space_points)
    k = 1/n_time_points
    return k/(h*h)

def coords(index, h):
    return index*h

def index(coords, h):
    return int(coords/h)

def euler_forward(bdry0: Callable[[float], float],
        bdry1: Callable[[float], float],
        u_initial: np.array,
        n_space_points: int,
        n_time_points: int):
    """
    Solve the heat equation via the forward euler scheme

    Parameters
    ----------
    bdry0 : function(t) -> u[0]
        A function capturing the boundary value at x=0 for a given time.
    bdry1 : function(t) -> u[1]
        A function capturing the boundary value at x=1 for a given time.
    u_initial : array_like
        The initial data from which the time evolution is to be computed.
        It should not contain the boundary points, since they are already given
        by bdry0 and bdry1.
    n_space_points : int
        The number of lattice points, excluding the boundary at x=0,x=1
    n_time_points : int
        The number of timesteps to be computed in the interval (0,1)

    Returns
    -------
    out: ndarray
        An numpy.ndarray object containing the temporal snapshots of u as
        columns.

    See also
    --------
    crank_nicolson : Solve the heat equation via the crank nicolson method
    euler_backward : Solve the heat equation via the backward euler scheme
    """
    h = 1./(n_space_points+1)           # space discretization constant
    k = 1./(n_time_points)              # time discretization constant
    r = k/(h*h)                         # parabolic mesh ratio
    u = np.zeros( (n_space_points, n_time_points+1) )
    u[:,0] = u_initial
    # main loop through the time steps
    for i in range(1, n_time_points):
        # deal with the boundary values first
        #u[0,i] = bdry0(k*(i-1)) - 2*u[0,i-1] + u[1,i-1]
        #u[-1,i] = bdry1(k*(i-1)) - 2*u[-1,i-1] + u[-2,i-1]
        logging.info(f"time step {i}")
        #for j in range(1, n_space_points-1):
        #    u[j,i] = u[j-1,i-1] - 2*u[j,i-1] + u[j+1,i-1]
        #u[:,i] *= r
        #u[:,i] += u[:,i-1]
        u[0,i] = u[0,i-1] + r*(bdry0(k*(i-1)) - 2*u[0,i-1] + u[1,i-1])
        u[-1,i] = u[-1,i-1] + r*(bdry1(k*(i-1)) - 2*u[-1,i-1] + u[-2,i-1])
        for j in range(1,n_space_points-1):
            u[j,i] = u[j,i-1] + r*(u[j-1,i-1] - 2*u[j,i-1] + u[j+1,i-1])
        if np.any(np.isnan(u[:,i])):
            logging.warning(f"nan encountered in u in step {i}")
    return u

def euler_backward(bdry0: Callable[[float], float],
        bdry1: Callable[[float], float],
        u_initial: np.array,
        n_space_points: int,
        n_time_points: int):
    """
    Solve the heat equation via the backward euler scheme

    Parameters
    ----------
    bdry0 : function(t) -> u[0]
        A function capturing the boundary value at x=0 for a given time.
    bdry1 : function(t) -> u[1]
        A function capturing the boundary value at x=1 for a given time.
    u_initial : array_like
        The initial data from which the time evolution is to be computed.
        It should not contain the boundary points, since they are already given
        by bdry0 and bdry1.
    n_space_points : int
        The number of lattice points, excluding the boundary at x=0,x=1
    n_time_points : int
        The number of timesteps to be computed in the interval (0,1)

    Returns
    -------
    out: ndarray
        An numpy.ndarray object containing the temporal snapshots of u as
        columns.

    See also
    --------
    crank_nicolson : Solve the heat equation via the crank nicolson method
    euler_forward : Solve the heat equation via the forward euler scheme
    """
    h = 1./(n_space_points+1)           # space discretization constant
    k = 1./(n_time_points)              # time discretization constant
    r = k/(h*h)                         # parabolic mesh ratio
    u = np.zeros( (n_space_points+2, n_time_points+1) )
    u[1:-1,0] = u_initial
    u[0,0], u[n_space_points,0] = bdry0(0), bdry1(0)
    M = lil_matrix( (n_space_points+2, n_space_points+2) )
    for i in range(n_space_points + 2):
        M[i,i] = 2*r+1

    for i in range(n_space_points + 1):
        M[i+1 ,i] = -r
        M[i,i+1] = -r
    M = csc_matrix(M)
    M_inv = inv(M)
    # main loop through the time steps
    for i in range(1, n_time_points):
        # deal with the boundary values first
        logging.info(f"euler_backward:Time step {i}/{n_time_points}")
        u[:,i] = M_inv.dot(u[:,i-1])
        u[0,i] = bdry0(i*k)
        u[-1,i] = bdry1(i*k)
    return u

def crank_nicolson_robin_bc(bdry_dirichlet_0: Callable[[float], float],
        bdry_dirichlet_1: Callable[[float], float],
        bdry_neumann_0: Callable[[float], float],
        bdry_neumann_1: Callable[[float], float],
        u_initial: np.array,
        n_space_points: int,
        n_time_points: int):
    """
    @param bdry0 A function that returns the boundary value at x=0
    @param bdry1 A function that returns the boundary value at x=1
    @param u_initial The (assumed to be consistent with bdry0 and bdry1) initial
        value of u. Those do not include the boundary points, since they are given by
        bdry0 and bdry1
    @param n_space_points the number of space points between x=0 and x=1
    @param n_time_points the number of time points on the interval including t=1
    """
    h = 1./(n_space_points+1)           # space discretization constant
    k = 1./(n_time_points)              # time discretization constant
    r = k/(h*h)                         # parabolic mesh ratio
    r_half = r*0.5
    u = np.zeros( (n_space_points+2, n_time_points+1) )
    u[1:-1,0] = u_initial
    u[0,0], u[n_space_points,0] = bdry0(0), bdry1(0)
    M = lil_matrix( (n_space_points+2, n_space_points+2) )
    B = lil_matrix( (n_space_points+2, n_space_points+2) )
    for i in range(n_space_points + 2):
        M[i,i] = r+1
        B[i,i] = 1-r

    for i in range(n_space_points + 1):
        M[i+1,i] = -r_half
        M[i,i+1] = -r_half
        B[i+1,i] = r_half
        B[i,i+1] = r_half
    M = csc_matrix(M)
    M_iter = inv(M) @ B
    # main loop through the time steps
    for i in range(1, n_time_points):
        # deal with the boundary values first
        logging.info(f"euler_backward:Time step {i}/{n_time_points}")
        u[:,i] = M_iter.dot(u[:,i-1])
        u[0,i] = bdry0(i*k)
        u[-1,i] = bdry1(i*k)
    return u

def crank_nicolson(bdry0: Callable[[float], float],
        bdry1: Callable[[float], float],
        u_initial: np.array,
        n_space_points: int,
        n_time_points: int):
    """
    Solve the heat equation via crank_nicolson (trapezoidal rule).

    Parameters
    ----------
    bdry0 : function(t) -> u[0]
        A function capturing the boundary value at x=0 for a given time.
    bdry1 : function(t) -> u[1]
        A function capturing the boundary value at x=1 for a given time.
    u_initial : array_like
        The initial data from which the time evolution is to be computed.
        It should not contain the boundary points, since they are already given
        by bdry0 and bdry1.
    n_space_points : int
        The number of lattice points, excluding the boundary at x=0,x=1
    n_time_points : int
        The number of timesteps to be computed in the interval (0,1)

    Returns
    -------
    out: ndarray
        An numpy.ndarray object containing the temporal snapshots of u as
        columns.

    See also
    --------
    euler_backward : Solve the heat equation via the backward euler scheme
    euler_forward : Solve the heat equation via the forward euler scheme
    """
    h = 1./(n_space_points+1)           # space discretization constant
    k = 1./(n_time_points)              # time discretization constant
    r = k/(h*h)                         # parabolic mesh ratio
    r_half = r*0.5
    u = np.zeros( (n_space_points+2, n_time_points+1) )
    u[1:-1,0] = u_initial
    u[0,0], u[n_space_points,0] = bdry0(0), bdry1(0)
    M = lil_matrix( (n_space_points+2, n_space_points+2) )
    B = lil_matrix( (n_space_points+2, n_space_points+2) )
    for i in range(n_space_points + 2):
        M[i,i] = r+1
        B[i,i] = 1-r

    for i in range(n_space_points + 1):
        M[i+1,i] = -r_half
        M[i,i+1] = -r_half
        B[i+1,i] = r_half
        B[i,i+1] = r_half
    M = csc_matrix(M)
    M_iter = inv(M) @ B
    # main loop through the time steps
    for i in range(1, n_time_points):
        # deal with the boundary values first
        logging.info(f"euler_backward:Time step {i}/{n_time_points}")
        u[:,i] = M_iter.dot(u[:,i-1])
        u[0,i] = bdry0(i*k)
        u[-1,i] = bdry1(i*k)
    return u

def integral_homogeneous(u: np.array, n_space_points: int):
    h = 1/(n_space_points+1)
    return np.sum(u)/h

def init_dirac_delta_function(n_space_points: int):
    u_init = np.zeros( n_space_points )
    u_init[n_space_points//2] = 1
    u_init *= integral_homogeneous(u_init, n_space_points)
    return u_init


def plot_animation(u: np.array,
        n_space_points: int,
        n_time_points: int,
        ymax: float = 1):
    """
    @param u The array containing the heat distributions for all timesteps. Its
        dimension is assumed to be (n_space_points, n_time_points)
    @param n_space_points The number of spacial grid points excluding the boundary
    @param n_time_points The number of time points given
    """
    fig, ax = plt.subplots()
    ln, = plt.plot([], [], 'b.')
    frames = u.transpose()

    def _init():
        ax.set_xlim(0,1)
        ax.set_ylim(0,ymax)
        return ln,

    def _update(frame):
        xdata = np.arange(0,1,1/(frame.size))
        ydata = frame
        ln.set_data(xdata,ydata)
        return ln,

    ani = FuncAnimation(fig, _update, interval=750, frames=frames, init_func=_init,
            blit=True)
    #ani.save("animation.mp4")
    plt.show()

class Parabolic_solver_1d():
    """
    A wrapper class for the one dimensional heat equation solvers.
    """
    def __init__(self, nsp, ntp, u_initial, bdry0, bdry1, bdry_type="dirichlet",
            solver=Solver_type.CN):
        self.n_space_points = nsp
        self.n_time_points = ntp
        if solver == Solver_type.EF:
            self.solver = euler_forward
        if solver == Solver_type.EB:
            self.solver = euler_backward
        if solver == Solver_type.CN:
            self.solver = crank_nicolson

        self.u_initial = u_initial
        self.bdry0 = lambda x: bdry0
        self.bdry1 = lambda x: bdry1
        self.bdry_type = bdry_type
        self.u = None

    def test(self):
        self.u = self.solver(self.bdry0, self.bdry1, self.u_initial,
                self.n_space_points, self.n_time_points)
        return self.u

    def plot(self):
        if self.u is None:
            logging.warning("Solver.plot called without executing the solve first")
            return
        ymax = np.max(self.u_initial)
        plot_animation(self.u, self.n_space_points, self.n_time_points, ymax=ymax)

def test():
    """
    general purpose testing method
    """
    # first we test euler_forward. here we can see that this method is highly
    # sensitive to the parabolic mesh ratio r. Change n_time_points to a lower
    # value and you will immediately see.
    print("testing euler_forward")
    n_space_points = 64
    n_time_points = 10000
    r = get_r(n_space_points, n_time_points)
    if r > 1:
        logging.warning(f"r={r} > 1 for {n_space_points} space and "+
                f"{n_time_points} time points")
    u_initial = np.zeros(n_space_points)
    u_initial[n_space_points//2] = 1
    bdry = lambda x: 0
    u = euler_forward(bdry, bdry, u_initial, n_space_points, n_time_points)
    #plot_animation(u, n_space_points, n_time_points)
    print("testing euler_forward complete")

    print("testing euler_backward")
    # next test euler_backward. here the parabolic mesh ratio should not affect
    # the stability of the method
    n_space_points = 512
    n_time_points = 1000
    u_initial = init_dirac_delta_function(n_space_points)
    ymax = u_initial.max()
    u = euler_backward(bdry, bdry, u_initial, n_space_points, n_time_points)
    #plot_animation(u, n_space_points+2, n_time_points, ymax=ymax)
    print("testing euler_backward complete")

    print("testing crank_nicolson")
    n_space_points = 128
    n_time_points = 10000
    u_initial = init_dirac_delta_function(n_space_points)
    ymax = u_initial.max()
    u = crank_nicolson(bdry, bdry, u_initial, n_space_points, n_time_points)
    plot_animation(u, n_space_points+2, n_time_points, ymax=ymax)
    print("testing crank_nicolson complete")

def main():
    # setup logging
    logging.basicConfig(filename="1d.log", filemode='w', level=logging.INFO)
    # test some stuff
    #test()
    u_init = init_dirac_delta_function(128)
    s = Parabolic_solver_1d(128, 10000, u_init, 0, 1)
    u = s.test()
    s.plot()

if __name__ == "__main__":
    main()
