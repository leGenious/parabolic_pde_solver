import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csc_matrix
from scipy.integrate import solve_ivp

# lex = x + y*nsp

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

def init_y0(n_space_points):
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

def plot_animation(u: np.array, n_space_points):
    fig, ax = plt.subplots()
    ln, = plt.imshow([])

    def _frames():
        for i in range( n_space_points ):
            yield u[:,i]

    def _init():
        return ln,

    def _update(frame):
        frame = frame.reshape(n_space_points, n_space_points)
        ln.set_data(frame)
        return ln,

    ani = FuncAnimation(fig, _update, interval=500, frames=frames,
            init_func=_init, blit=True)
    plt.show()

def plot(n_space_points, y):
    y = y.reshape(n_space_points, n_space_points)
    plt.imshow(y)
    plt.show()

def method_of_lines():
    n_space_points = 64
    ode = MOL(n_space_points)
    plt.spy(ode.M, markersize=2)
    plt.show()
    y0 = init_y0(n_space_points)
    plot(n_space_points,y0)
    result = solve_ivp(ode, (0,1), y0, method='RK45')
    for i in range( len(result.t) ):
        plot(n_space_points, result.y[:,i])

def main():
    method_of_lines()

if __name__ == "__main__":
    main()
