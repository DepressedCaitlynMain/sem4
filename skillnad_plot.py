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

# Function to calculate the RMS error
def calculate_error(grid, prev_grid):
    error = np.sqrt(np.sum((grid[1:-1, 1:-1] - prev_grid[1:-1, 1:-1])**2) / grid[1:-1, 1:-1].size)
    return error

# Function to initialize the grid with custom boundary conditions
def initialize_grid_with_custom_boundary(n, left=5, top=10, right=5, bottom=10, center_value=0):
    v = np.full((n + 2, n + 2), center_value)  # Include boundary points
    v[:, 0] = left   # Left boundary
    v[0, :] = top    # Top boundary
    v[:, -1] = right # Right boundary
    v[-1, :] = bottom # Bottom boundary
    return v

# Function to run relaxation for both Jacobi and Gauss-Seidel methods
def run_relaxation(method, grid_size, left=5, top=10, right=5, bottom=10, tolerance=1e-3, nsteps=10000):
    n = grid_size
    v = initialize_grid_with_custom_boundary(n, left, top, right, bottom)
    v_new = v.copy()  # For Jacobi method

    prev_grid = v.copy()  # To calculate error and detect convergence

    for _ in range(nsteps):
        prev_grid[:] = v
        if method == relax_normal:  # Jacobi method
            relax_normal(v, v_new, n)
            v[1:-1, 1:-1] = v_new[1:-1, 1:-1]
        elif method == relax_gauss_seidel:  # Gauss-Seidel method
            relax_gauss_seidel(v, n)
        
        error = calculate_error(v, prev_grid)
        if error < tolerance:
            break

    return v  # Return the final grid

if __name__ == "__main__":
    grid_sizes = list(range(10, 90, 5))
    max_differences = []

    # Iterate over grid sizes
    for grid_size in grid_sizes:
        # Run Jacobi method
        final_grid_jacobi = run_relaxation(relax_normal, grid_size, tolerance=1e-3)

        # Run Gauss-Seidel method
        final_grid_gauss_seidel = run_relaxation(relax_gauss_seidel, grid_size, tolerance=1e-3)

        # Compute the maximum difference between the two grids
        max_difference = np.max(np.abs(final_grid_jacobi - final_grid_gauss_seidel))
        max_differences.append(max_difference)

    # Plot the maximum differences vs grid sizes
    plt.plot(grid_sizes, max_differences, marker='o', label="Max Difference (Default vs Gauss-Seidel)")
    plt.title("Maximum Differences Between Final Grids")
    plt.xlabel("Grid Size (n)")
    plt.ylabel("Maximum Difference")
    plt.legend()
    plt.grid(True)
    plt.show()
