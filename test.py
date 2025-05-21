import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Demand points and weights
np.random.seed(10)
points = np.array([[1, 3], [4, 4], [5, 1], [7, 3], [3, 1], [3, 5], [5, 3]])
weights = np.array([1, 3.0, 3.5, 1, 2.5, 5, 2])

# Weiszfeld algorithm with trajectory
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
ax.set_ylim(-2, 8)  # Extend y-limits to leave room for text
ax.set_xticks(np.arange(0, 9, 1))
ax.set_yticks(np.arange(0, 9, 1))
ax.set_title('Weiszfeld Method - Single Facility Location Problem', fontsize=14, fontweight='bold')
ax.axis('equal')
ax.axis('off')

# Demand points
ax.scatter(points[:, 0], points[:, 1], c='blue', s=100, label='Demand Points')
for i, (px, py) in enumerate(points):
    ax.text(px + 0.1, py + 0.1, f'w={weights[i]}', fontsize=9)

# Trajectory line and optimal marker
trajectory_line, = ax.plot([], [], 'r--o', label='Weiszfeld Iterations')
opt_point = ax.scatter([], [], c='green', s=150, marker='X', label='Optimal Location')

# Lines from current point to demand points
connection_lines = [ax.plot([], [], 'gray', lw=1, alpha=0.6)[0] for _ in points]

# Cost text
cost_text = ax.text(4, -0.2, '1234', fontsize=12, ha='center', va='center', fontweight='bold', color='black')

# Legend
ax.legend()

# Update function for animation
def update(frame):
    path = trajectory[:frame + 1]
    current_point = path[-1]

    # Update trajectory line
    trajectory_line.set_data(path[:, 0], path[:, 1])

    # Update optimal point marker
    if frame == len(trajectory) - 1:
        opt_point.set_offsets([current_point])
    else:
        opt_point.set_offsets([[-10, -10]])  # Hide in intermediate frames

    # Update connection lines
    for line, demand_point in zip(connection_lines, points):
        line.set_data([current_point[0], demand_point[0]], [current_point[1], demand_point[1]])

    # Compute and display total cost
    dists = np.linalg.norm(points - current_point, axis=1)
    total_cost = np.sum(weights * dists)

    # Update cost text
    if frame == len(trajectory) - 1:
        cost_text.set_color('green')
    else:
        cost_text.set_color('black')
    cost_text.set_text(f'Total Weighted Distance: {total_cost:.3f}')

    return [trajectory_line, opt_point, cost_text] + connection_lines

# Create animation
ani = FuncAnimation(fig, update, frames=len(trajectory), interval=500, repeat=True)

plt.tight_layout()
plt.show()
