import numpy as np
import matplotlib.pyplot as plt

# Function to relax the grid using Jacobi method (normal updates)
def relax_normal(grid, grid_new, n):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            grid_new[i, j] = 0.25 * (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1])
# Function to relax the grid using Gauss-Seidel method (sequential updates)
def relax_gauss_seidel(v, n):
    """
    Update the potential grid using the Gauss-Seidel method.
    Updates are applied sequentially, immediately using the most recent values.
    """
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            v[i, j] = 0.25 * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1])
    
def calculate_error_gauss_seidel(v, n):
    """
    Compute the maximum error based on the difference between old and new values.
    """
    max_error = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            updated_value = 0.25 * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1])
            error = abs(updated_value - v[i, j])
            #print(error)
            max_error = max(max_error, error)
    return max_error
def relax_checkerboard(v, n):
    """
    Update the grid using the checkerboard method:
    - Update red sites first (i + j is even).
    - Update black sites next (i + j is odd).
    """
    # Update red sites
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if (i + j) % 2 == 0:  # Red sites
                v[i, j] = 0.25 * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1])

    # Update black sites
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if (i + j) % 2 == 1:  # Black sites
                v[i, j] = 0.25 * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1])

def calculate_error(v,new_v):
    return np.max(np.abs(new_v[1:-1, 1:-1] - v[1:-1, 1:-1]))
# Function to initialize the grid with custom boundary conditions
def initialize_grid_with_custom_boundary(n, left=5, top=10, right=5, bottom=10, center_value=0):
    v = np.zeros((n + 2, n + 2))  # Include boundary points
    v[:, 0] = left   # Left boundary
    v[0, :] = top    # Top boundary
    v[:, -1] = right # Right boundary
    v[-1, :] = bottom # Bottom boundary

    v[1:-1, 1:-1] = 7.5


    return v


# Function to run the relaxation for both Jacobi and Gauss-Seidel methods
def run_relaxation_gauss(v, grid_size, left=10, top=10, right=10, bottom=0, tolerance=1e-3, nsteps=10000):
    n = grid_size
    
    #v_new = v.copy()  # For Jacobi method

    #prev_grid = v.copy()  # To calculate error and detect convergence
    iterations = 0
    while True:
        """prev_grid[:] = v
        if method == relax_normal:  # Jacobi method
            relax_normal(v, v_new, n)
            error = calculate_error(v_new, v)
            v[1:-1, 1:-1] = v_new[1:-1, 1:-1]
            

        elif method == relax_gauss_seidel:  # Gauss-Seidel method"""
        max_error = calculate_error_gauss_seidel(v, n)
        #print(max_error)
        if max_error < tolerance:
            break
        relax_gauss_seidel(v, n)
        iterations += 1

    print("ok")
    return iterations,v
def run_relaxation_checkerboard(v, grid_size, left=10, top=10, right=10, bottom=0, tolerance=1e-3, nsteps=10000):
    n = grid_size
    
    #v_new = v.copy()  # For Jacobi method

    #prev_grid = v.copy()  # To calculate error and detect convergence
    iterations = 0
    while True:

        max_error = calculate_error_gauss_seidel(v, n)
        #print(max_error)
        if max_error < tolerance:
            break
        relax_checkerboard(v, n)
        iterations += 1

    print("ok")
    return iterations,v

def run_relaxation_normal(v, grid_size, left=10, top=10, right=10, bottom=0, tolerance=1e-3, nsteps=10000):
    n = grid_size
    
    v_new = v.copy()  # For Jacobi method

    #prev_grid = v.copy()  # To calculate error and detect convergence
    iterations = 0
    while True:
        relax_normal(v, v_new, n)
        error = calculate_error(v_new, v)
        v[1:-1, 1:-1] = v_new[1:-1, 1:-1]
            


        if error < tolerance:
            break
        iterations += 1

    print("ok")
    return iterations,v

if __name__ == "__main__":
    grid_sizes = list(range(10, 300, 20))
    jacobi_iterations = []
    gauss_seidel_iterations = []
    checker_iterations = []

    max_differences = []
    # Collect iterations for Jacobi method
    for grid_size in grid_sizes:
        v = initialize_grid_with_custom_boundary(grid_size, left = 10, top = 10, right = 10, bottom = 0)
        iterations,final_v_normal = run_relaxation_normal(v, grid_size, tolerance=0.01)
        jacobi_iterations.append(iterations)
        v = initialize_grid_with_custom_boundary(grid_size, left = 10, top = 10, right = 10, bottom = 0)
        
    # Collect iterations for Gauss-Seidel method
        iterations,final_v_gauss = run_relaxation_gauss(v, grid_size, tolerance=0.01)
        gauss_seidel_iterations.append(iterations)

        #checkboard

        v = initialize_grid_with_custom_boundary(grid_size, left = 10, top = 10, right = 10, bottom = 0)
        iterations,final_v_checkerboard = run_relaxation_checkerboard(v, grid_size, tolerance=0.01)
        checker_iterations.append(iterations)


        # Compute the maximum difference between the two grids
        max_difference = np.max(np.abs(final_v_checkerboard - final_v_gauss))
        max_differences.append(max_difference)


    # Plot the results
    plt.figure(1)
    plt.plot(grid_sizes, jacobi_iterations, label="Default Method", marker='o')
    plt.plot(grid_sizes, gauss_seidel_iterations, label="Gauss-Seidel Method", marker='s')
    plt.plot(grid_sizes, checker_iterations, label="Checkerboard Method", marker='s')

    plt.title("Iterations vs Grid Size")
    plt.xlabel("Grid Size (n)")
    plt.ylabel("Iterations to Converge")

    plt.legend()
    plt.grid(True)

    plt.figure(2)
    plt.title("Maximum Differences Between Final Grids")

    plt.xlabel("Grid Size (n)")
    plt.ylabel("Difference between Gauss and checkerboard methods")
    plt.plot(grid_sizes,max_differences, label = "max(abs(difference of the grid of the two methods))",marker = "o")
    plt.legend()
    plt.grid(True)
    plt.show()
