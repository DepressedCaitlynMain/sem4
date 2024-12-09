import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to relax the grid
def relax(grid, grid_new, n):
    """
    Perform one step of relaxation using the Jacobi method.
    """
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            grid_new[i, j] = 0.25 * (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1])

# Function to initialize the grid with custom boundary conditions
def initialize_grid_with_custom_boundary(n, left=5, top=10, right=5, bottom=10, center_value=0):
    """
    Initialize the grid with specified boundary values and optional center value.
    """
    v = np.full((n + 2, n + 2), center_value)  # Include boundary points
    vnew = np.ones_like(v)*1000

    # Set boundary conditions
    v[:, 0] = left    # Left boundary
    v[0, :] = top     # Top boundary
    v[:, -1] = right  # Right boundary
    v[-1, :] = bottom # Bottom boundary

    return v, vnew

# Function to calculate the maximum error between two grids
def calculate_error(grid, grid_new):
    """
    Compute the maximum difference between the new and old grid values.
    """
    return np.max(np.abs(grid_new[1:-1, 1:-1] - grid[1:-1, 1:-1]))

# Function to animate relaxation
def animate_relaxation(n, left=5, top=10, right=5, bottom=10, tolerance=0.01, nsteps=1000):
    """
    Perform Jacobi relaxation on the grid and animate the process.
    """
    # Initialize grid and new grid
    v, vnew = initialize_grid_with_custom_boundary(n, left, top, right, bottom)
    
    # Set up the figure for animation
    fig, ax = plt.subplots()
    im = ax.imshow(v, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(im, ax=ax, label="Potential")
    ax.set_title(f"Relaxation Animation\nBoundary: Left={left}, Top={top}, Right={right}, Bottom={bottom}")
    error = 10
    def update(frame):
        nonlocal v, vnew, error
        if frame > 0:
            relax(v, vnew, n)  # Perform relaxation step
            v[1:-1, 1:-1] = vnew[1:-1, 1:-1]  # Update interior points
            
            # Stop the animation if converged
            tolerance = 0.01
            if  error< tolerance:
                print(f"Converged at frame {frame}")
                #anim.event_source.stop()
            error = calculate_error(v, vnew)
        im.set_array(v)
        return [im]

    anim = FuncAnimation(fig, update, frames=nsteps + 1, interval=100, blit=True, repeat=False)
    plt.show()

# Main execution
if __name__ == "__main__":
    n = 20  # Grid size for higher resolution

    # Case 1: Boundary values of 5, 10, 5, and 10
    print("Animating Case 1: Boundary values 5, 10, 5, 10")
    animate_relaxation(n, left=5, top=10, right=5, bottom=10)

    # Case 2: Boundary values of 10, 10, 10, and 0
    print("Animating Case 2: Boundary values 10, 10, 10, 0")
    animate_relaxation(n, left=10, top=10, right=10, bottom=0)
