import numpy as np
import matplotlib.pyplot as plt

def random_walk(grid, x0, y0,n):
    x, y = x0, y0
    
    while 0 < x < (n-1) and 0 < y < n - 1:  # Stay within the interior of the grid
        step = np.random.choice(["up", "down", "left", "right"])
        if step == "up":
            x -= 1
        elif step == "down":
            x += 1
        elif step == "left":
            y -= 1
        elif step == "right":
            y += 1
    return x, y

def calculate_potential(green_function, boundary_potential):
    """
    Use the Green's function G to calculate the potential V(x, y).
    """
    n, m = green_function.shape[:2]
    potential = np.zeros((n, m), dtype=float)

    for x in range(1, n - 1):
        for y in range(1, m - 1):
            # Sum over all boundary points
            for xb in range(n):
                for yb in range(m):
                    if xb == 0 or xb == n - 1 or yb == 0 or yb == m - 1:  # Boundary points
                        potential[x, y] += green_function[x, y, xb, yb] * boundary_potential[xb, yb]

    return potential

def initialize_boundary_potential(n, left=5, top=10, right=5, bottom=10):
    """
    Initialize a boundary potential with fixed values on the boundary.
    """
    v = np.zeros((n + 2, n + 2), dtype=float)
    v[:, 0] = left   # Left boundary
    v[0, :] = top    # Top boundary
    v[:, -1] = right # Right boundary
    v[-1, :] = bottom # Bottom boundary
    return v

def compute_greens_function(grid, start_x, start_y, num_walks):
    n, m = grid.shape
    G = np.zeros_like(grid)  
    for _ in range(num_walks):
        boundary_x, boundary_y = random_walk(grid, start_x, start_y,n)
        G[boundary_x, boundary_y] += 1  
    return G / np.sum(G)  

if __name__ == "__main__":
    n = 9  # Grid size
    num_walks = 200  # Number of random walkers
    grid = np.zeros((n + 2, n + 2))  # Grid to define geometry

    # Initialize boundary potential
    boundary_potential = initialize_boundary_potential(n)

    # Step 1: Compute the Green's function
    print("Computing Green's function...")
    potential = np.zeros_like(grid)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            G = compute_greens_function(grid, i, j, num_walks)
            # Compute the potential as a weighted sum of boundary probabilities
            potential[i, j] = (
                np.sum(G[0, :]) * 10 +  # Top boundary
                np.sum(G[-1, :]) * 10 +  # Bottom boundary
                np.sum(G[:, 0]) * 5 +   # Left boundary
                np.sum(G[:, -1]) * 5    # Right boundary
            )
    potential[:, 0] = 5   # Left boundary
    potential[0, :] = 10    # Top boundary
    potential[:, -1] = 5 # Right boundary
    potential[-1, :] = 10 # Bottom boundary

    # Step 3: Visualize the potential
    plt.figure()
    plt.title("Greens Potential")
    plt.imshow(potential, origin='upper', cmap='viridis')
    plt.colorbar(label="Potential")
    plt.grid(False)
    plt.show()
