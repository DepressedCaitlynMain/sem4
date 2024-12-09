import numpy as np
import matplotlib.pyplot as plt

# Function to relax the grid using Jacobi method (normal updates)
def relax_normal(grid, grid_new, n):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            grid_new[i, j] = 0.25 * (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1])

# Function to relax the grid using Gauss-Seidel method (sequential updates)
def relax_gauss_seidel(v, n):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            v[i, j] = 0.25 * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1])

# Function to relax the grid using the checkerboard method
def relax_checkerboard(v, n):
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

# Function to calculate the maximum error
def calculate_error(grid, prev_grid):
    return np.max(np.abs(prev_grid[1:-1, 1:-1] - grid[1:-1, 1:-1]))

def calculate_error_checkerboard_gauss(v, n):
    """
    Compute the maximum error for the grid.
    """
    max_error = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            updated_value = 0.25 * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1])
            error = abs(updated_value - v[i, j])
            max_error = max(max_error, error)
    return max_error


# Function to initialize the grid with custom boundary conditions
def initialize_grid_with_custom_boundary(n, left=5, top=10, right=5, bottom=10, center_value=7.5):
    v = np.full((n + 2, n + 2), center_value)  # Include boundary points
    v[:, 0] = left   # Left boundary
    v[0, :] = top    # Top boundary
    v[:, -1] = right # Right boundary
    v[-1, :] = bottom # Bottom boundary
    return v

# Function to run the relaxation for any method
def run_relaxation(method, grid_size, left=10, top=10, right=0, bottom=10, tolerance=1e-3, nsteps=10000):
    n = grid_size
    v = initialize_grid_with_custom_boundary(n, left, top, right, bottom)
    prev_grid = v.copy()  # To calculate error and detect convergence
    iterations = 0


    for _ in range(nsteps):
        prev_grid[:] = v
        method(v, n)  # Run the relaxation method
        error = calculate_error_checkerboard_gauss(v, n)
        iterations += 1

        if error < tolerance:
            break

    return iterations

if __name__ == "__main__":
    grid_sizes = list(range(10, 400, 90))
    gauss_seidel_iterations = []
    checker_iter = []

    # Collect iterations for Gauss-Seidel method
    for grid_size in grid_sizes:
        iterations = run_relaxation(relax_gauss_seidel, grid_size, tolerance=1e-3)
        gauss_seidel_iterations.append(iterations)

    # Collect iterations for Checkerboard method
    for grid_size in grid_sizes:
        iterations = run_relaxation(relax_checkerboard, grid_size, tolerance=1e-3)
        checker_iter.append(iterations)

    # Plot the results
    plt.plot(grid_sizes, gauss_seidel_iterations, label="Gauss-Seidel Method", marker='o')
    plt.plot(grid_sizes, checker_iter, label="Checkerboard Method", marker='s')

    plt.title("Iterations vs Grid Size")
    plt.xlabel("Grid Size (n)")
    plt.ylabel("Iterations to Converge")
    plt.legend()
    plt.grid(True)
    plt.show()
