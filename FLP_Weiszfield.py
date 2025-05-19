import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Demand points and weights
np.random.seed(10)
points = np.array([[1, 3], [4, 4], [5, 1], [7, 3], [3, 1], [3, 5], [5, 3]])
weights = np.array([1, 3.0, 3.5, 1, 2.5, 5, 2])

def weiszfeld(points, weights, tol=1e-6, max_iter=20):
    trajectory = []
    x = np.array([1., 1.])
    trajectory.append(x.copy())

    for _ in range(max_iter):
        numer = np.zeros_like(x)
        denom = 0.0
        for i, p in enumerate(points):
            dist = np.linalg.norm(x - p)
            if dist < 1e-12:
                return p, trajectory
            numer += weights[i] * p / dist
            denom += weights[i] / dist
        x_new = numer / denom
        trajectory.append(x_new.copy())
        if np.linalg.norm(x - x_new) < tol:
            break
        x = x_new

    return x, trajectory

# Run Weiszfeld algorithm
optimal_location, trajectory = weiszfeld(points, weights)
trajectory = np.array(trajectory)

# Plot setup
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('lightyellow')
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.set_xticks(np.arange(0, 9, 1))
ax.set_yticks(np.arange(0, 9, 1))
ax.set_title('Weiszfeld Method - Single Facility Location Problem', fontsize=14, fontweight='bold')
ax.axis('equal')
ax.axis('off')

# Demand points
ax.scatter(points[:, 0], points[:, 1], c='blue', s=100, label='Demand Points')
for i, (px, py) in enumerate(points):
    ax.text(px + 0.1, py + 0.1, f'w={weights[i]}', fontsize=9)

# Trajectory and optimal marker
trajectory_line, = ax.plot([], [], 'r--o', label='Weiszfeld Iterations')
opt_point = ax.scatter([], [], c='green', s=150, marker='X', label='Optimal Location')

ax.legend()

# Update function
def update(frame):
    path = trajectory[:frame + 1]
    trajectory_line.set_data(path[:, 0], path[:, 1])
    if frame == len(trajectory) - 1:
        opt_point.set_offsets([path[-1]])
    else:
        opt_point.set_offsets([[-10, -10]])  # Hide
    return trajectory_line, opt_point

# Prevent garbage collection by storing ani in a global variable
ani = FuncAnimation(fig, update, frames=len(trajectory), interval=500, repeat=True)

plt.tight_layout()
plt.show()
