import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation



# checker=1: no checkboard, checker=2: checkerboard (note: n should be even)
checker = 1

# perform one step of relaxation
def relax(n, v, vnew):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            vnew[i, j] = 0.25 * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1])

def calc_error(v,new_v):
    return np.max(np.abs(new_v[1:-1, 1:-1] - v[1:-1, 1:-1]))

def excersice_42a():
    global vnew
    itterations = 0
    tolerance = 0.01
    error = tolerance+1 #just something
    while error > tolerance:
        #print(itterations)
        relax(n,v,vnew)
        error=calc_error(v,vnew)
        v[1:-1, 1:-1] = vnew[1:-1, 1:-1]
        itterations+=1

    print(f"itterations for size n = {n}: {itterations}")

#Create the given matrix and a zero matrix of same size
n = 9
v = np.ones((n+2, n+2))*9 #set entire matrix to 9
vnew = np.zeros((n+2, n+2))
L = 10

# Set the boundary conditions
v[0, :] = 10  # Top boundary
v[-1, :] = 10  # Bottom boundary
v[:, 0] = 10  # Left boundary
v[:, -1] = 10  # Right boundary

excersice_42a()

#dubble gridsize
n = 18
v = np.ones((n+2, n+2))*9 #set entire matrix to 9
vnew = np.zeros((n+2, n+2))
L = 10

# Set the boundary conditions
v[0, :] = 10  # Top boundary
v[-1, :] = 10  # Bottom boundary
v[:, 0] = 10  # Left boundary
v[:, -1] = 10  # Right boundary

excersice_42a()
#print(v)



