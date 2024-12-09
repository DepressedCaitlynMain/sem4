import numpy as np
import matplotlib.pyplot as plt

# Function to initialize the grid with custom boundary conditions
def initialize_grid_with_custom_boundary(n, left=5, top=10, right=5, bottom=10, center_value=7.5):
    v = np.full((n + 2, n + 2), center_value)  # Include boundary points
    v[:, 0] = left   # Left boundary
    v[0, :] = top    # Top boundary
    v[:, -1] = right # Right boundary
    v[-1, :] = bottom # Bottom boundary
    return v

# Gauss-Seidel relaxation function
def relax_gauss_seidel(v, n):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            v[i, j] = 0.25 * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1])

# Function to solve Laplace's equation numerically
def solve_laplace_gauss_seidel(n, tolerance=1e-3, max_iterations=10000):
    v = initialize_grid_with_custom_boundary(n)
    iterations = 0
    while iterations < max_iterations:
        old_v = v.copy()
        relax_gauss_seidel(v, n)
        max_error = np.max(np.abs(v - old_v))
        iterations += 1
        if max_error < tolerance:
            break
    print(f"Gauss-Seidel converged in {iterations} iterations.")
    return v

# Random walk simulation functions
def random_walk(grid, x0, y0):
    x, y = x0, y0
    n, m = grid.shape
    while 0 < x < n - 1 and 0 < y < m - 1:  # Stay within the interior of the grid
        step = np.random.choice(["up", "down", "left", "right"])
        if step == "up":
            x -= 1
        elif step == "down":
            x += 1
        elif step == "left":
            y -= 1
        elif step == "right":
            y += 1
    return grid[x, y]

def estimate_potential(grid, start_x, start_y, num_walks):
    boundary_values = [random_walk(grid, start_x, start_y) for _ in range(num_walks)]
    return np.mean(boundary_values)

def run_random_walks(n, num_walks):
    grid = initialize_grid_with_custom_boundary(n)
    n, m = grid.shape
    potential = grid.copy()  # Start with the boundary conditions in the potential grid
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            potential[i, j] = estimate_potential(grid, i, j, num_walks)
    return potential

# Main comparison script
if __name__ == "__main__":
    n = 20  # Grid size (interior points)

    # Solve Laplace's equation using the Gauss-Seidel method
    gauss_seidel_solution = solve_laplace_gauss_seidel(n)

    # Solve Laplace's equation using the random walk method
    random_walk_solution_100 = run_random_walks(n, 100)
    random_walk_solution_1000 = run_random_walks(n, 1000)

    # Plot Gauss-Seidel solution
    plt.figure(1)
    plt.title("Gauss-Seidel Solution")
    plt.imshow(gauss_seidel_solution, origin='upper', cmap='viridis')
    plt.colorbar(label="Potential")
    plt.grid(False)

    # Plot Random Walk solution with 100 walks
    plt.figure(2)
    plt.title("Random Walk Solution (100 Walks)")
    plt.imshow(random_walk_solution_100, origin='upper', cmap='viridis')
    plt.colorbar(label="Potential")
    plt.grid(False)

    # Plot Random Walk solution with 1000 walks
    plt.figure(3)
    plt.title("Random Walk Solution (1000 Walks)")
    plt.imshow(random_walk_solution_1000, origin='upper', cmap='viridis')
    plt.colorbar(label="Potential")
    plt.grid(False)

    # Show all plots
    plt.show()
