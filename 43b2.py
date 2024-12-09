import numpy as np
import matplotlib.pyplot as plt

# Random walk function as before
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

# Modified to return boundary values for variance analysis
def estimate_potential_and_variance(grid, start_x, start_y, num_walks):
    boundary_values = [random_walk(grid, start_x, start_y) for _ in range(num_walks)]
    return np.mean(boundary_values), np.var(boundary_values)

def initialize_grid_with_custom_boundary(n, left=5, top=10, right=5, bottom=10, center_value=7.5):
    v = np.full((n + 2, n + 2), center_value)  # Include boundary points
    v[:, 0] = left   # Left boundary
    v[0, :] = top    # Top boundary
    v[:, -1] = right # Right boundary
    v[-1, :] = bottom # Bottom boundary
    return v

def convergence_analysis(grid, start_x, start_y, walk_counts):
    means, variances = [], []
    mse = []
    prev_pot = 0
    for num_walks in walk_counts:
        mean, var = estimate_potential_and_variance(grid, start_x, start_y, num_walks)
        means.append(mean)
        variances.append(var)
        mse.append(np.mean((mean-prev_pot))**2)
        prev_pot = mean
    return means, variances,mse


if __name__ == "__main__":
    n = 10  # Grid size
    grid = initialize_grid_with_custom_boundary(n)

    # Points for analysis: near-surface and center
    near_surface_point = (1, 1)
    center_point = (n // 2, n // 2)

    # Number of walkers to test
    walk_counts = [1, 10, 100, 1000, 10**4]

    # Analyze convergence at near-surface point
    #near_surface_means, near_surface_vars = convergence_analysis(grid, *near_surface_point, walk_counts)

    #find mse


    # Analyze convergence at the center point
    center_means, center_vars,mse = convergence_analysis(grid, *center_point, walk_counts)

    # Plot variance as a function of number of walks
    plt.figure()
    plt.loglog(walk_counts, mse,marker = "x", label="MSE near surface point")
    plt.xlabel("Number of Walks (log scale)")
    plt.ylabel("MSE")
    plt.title("MSE Convergence")
    plt.legend()
    plt.grid(True)

    plt.show()
