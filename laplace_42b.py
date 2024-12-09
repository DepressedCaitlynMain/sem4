import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Function to relax the grid
def relax(grid, grid_new, n):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            grid_new[i, j] = 0.25 * (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1])

# Function to calculate the maximum error
def calculate_error(grid, grid_new, n):
    return np.max(np.abs(grid_new[1:-1, 1:-1] - grid[1:-1, 1:-1]))

# Function to initialize the grid with custom boundary conditions
def initialize_grid_with_custom_boundary(n, left=5, top=10, right=5, bottom=10, center_value=0):
    v = np.full((n + 2, n + 2), center_value)  # Include boundary points
    vnew = np.zeros_like(v)

    # Set boundary conditions
    v[:, 0] = left   # Left boundary
    v[0, :] = top    # Top boundary
    v[:, -1] = right # Right boundary
    v[-1, :] = bottom # Bottom boundary

    return v, vnew

# Perform the relaxation method with animation
def animate_relaxation(v, vnew, n, tolerance=0.01, nsteps=1000):
    fig, ax = plt.subplots()
    im = ax.imshow(v, cmap='hot', interpolation='nearest', origin='lower')
    ax.set_title("Relaxation Animation")
    plt.colorbar(im, ax=ax, label="Potential")

    def update(frame):
        nonlocal v, vnew
        if frame > 0:
            relax(v, vnew, n)
            v[1:-1, 1:-1] = vnew[1:-1, 1:-1]
        im.set_array(v)
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=nsteps + 1, interval=200, blit=True, repeat=False)
    plt.show()

    return v

if __name__ == "__main__":
    n = 20  # Grid size for higher resolution

    # Case 1: Boundary values of 5, 10, 5, and 10
    v1, vnew1 = initialize_grid_with_custom_boundary(n, left=5, top=10, right=5, bottom=10)
    print("Animating Case 1: Boundary values 5, 10, 5, 10")
    final_grid1 = animate_relaxation(v1, vnew1, n)
    
    plt.figure()  # Set the figure size
    plt.imshow(final_grid1, cmap='viridis', interpolation='nearest', origin='upper')
    plt.colorbar(label="Value")  # Add a colorbar to indicate the value scale
    plt.show()
    # Case 2: Boundary values of 10, 10, 10, and 0
    v2, vnew2 = initialize_grid_with_custom_boundary(n, left=10, top=10, right=10, bottom=0)
    print("Animating Case 2: Boundary values 10, 10, 10, 0")
    final_grid2 = animate_relaxation(v2, vnew2, n)
