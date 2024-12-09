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

def estimate_potential(grid, start_x, start_y, num_walks):
    boundary_values = [random_walk(grid, start_x, start_y) for _ in range(num_walks)]
    return np.mean(boundary_values)

# Function to initialize the grid with custom boundary conditions
def initialize_grid_with_custom_boundary(n, left=5, top=10, right=5, bottom=10, center_value=7.5):
    v = np.full((n + 2, n + 2), center_value)  # Include boundary points
    v[:, 0] = left   # Left boundary
    v[0, :] = top    # Top boundary
    v[:, -1] = right # Right boundary
    v[-1, :] = bottom # Bottom boundary
    return v

def run_random_walks(n, num_walks):
    # Initialize grid with boundary conditions
    grid = initialize_grid_with_custom_boundary(n)
    n, m = grid.shape
    potential = grid.copy()  # Start with the boundary conditions in the potential grid

    # Calculate potential for interior points using random walks
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            potential[i, j] = estimate_potential(grid, i, j, num_walks)

    return potential

if __name__ == "__main__":
    n = 10

    # Perform random walks to estimate potential
    potential100 = run_random_walks(n, 100)
    potential1000 = run_random_walks(n, 1000)

    # Plot the results for 100 walks
    plt.figure(1)
    plt.title("100 Walks")
    plt.imshow(potential100, origin='upper', cmap='viridis')
    plt.colorbar(label="Potential")
    plt.grid(False)

    # Plot the results for 1000 walks
    plt.figure(2)
    plt.title("1000 Walks")
    plt.imshow(potential1000, origin='upper', cmap='viridis')
    plt.colorbar(label="Potential")
    plt.grid(False)

    plt.show()
