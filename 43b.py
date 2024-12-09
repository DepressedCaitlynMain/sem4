import numpy as np
import matplotlib.pyplot as plt

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

def estimate_potential_and_variance(grid, start_x, start_y, num_walks):
    boundary_values = [random_walk(grid, start_x, start_y) for _ in range(num_walks)]
    mean_potential = np.mean(boundary_values)
    variance = np.var(boundary_values)
    return mean_potential, variance

def initialize_grid_with_custom_boundary(n, left=5, top=10, right=5, bottom=10, center_value=7.5):
    v = np.full((n + 2, n + 2), center_value)  # Include boundary points
    v[:, 0] = left   # Left boundary
    v[0, :] = top    # Top boundary
    v[:, -1] = right # Right boundary
    v[-1, :] = bottom # Bottom boundary
    return v

# Analyze convergence for various points
def analyze_convergence_with_variance(n, points, walkers_list):
    grid = initialize_grid_with_custom_boundary(n)
    results = {point: {"potentials": [], "variances": []} for point in points}
    
    for num_walkers in walkers_list:
        for point in points:
            x, y = point
            potential, variance = estimate_potential_and_variance(grid, x, y, num_walkers)
            results[point]["potentials"].append(potential)
            results[point]["variances"].append(variance)
    
    return results

# Plot variance as a function of walkers
def plot_variance_convergence(points, walkers_list, results):
    plt.figure(1)

    for point in points:
        plt.plot(walkers_list, results[point]["variances"], label=f"Point {point}")
    plt.title("Variance of Potential Estimate vs Number of Walkers")
    plt.xlabel("Number of Walkers")
    plt.ylabel("Variance")
    plt.yscale("log")  # Use a logarithmic scale for better visualization
    plt.legend()
    plt.grid(True)

    plt.figure(2)
    for point in points:
        plt.plot(walkers_list, results[point]["potentials"], label=f"Point {point}")
    plt.title("Result of Potential Estimate vs Number of Walkers")
    plt.xlabel("Number of Walkers")
    plt.ylabel("Potential")
    plt.yscale("log")  # Use a logarithmic scale for better visualization
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    n = 10  # Grid size
    points = [(5, 5), (5, 4), (5, 3), (5, 2), (5, 1)]  # Points to analyze
    walkers_list = [100, 500, 1000, 5000, 10000]  # Different numbers of walkers

    # Analyze convergence
    results = analyze_convergence_with_variance(n, points, walkers_list)

    # Plot variance convergence
    plot_variance_convergence(points, walkers_list, results)
