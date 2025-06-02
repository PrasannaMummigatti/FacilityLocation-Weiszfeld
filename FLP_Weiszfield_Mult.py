import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

def assign_points_to_facilities(points, facilities):
    dists = np.linalg.norm(points[:, np.newaxis] - facilities, axis=2)
    return np.argmin(dists, axis=1)

def weiszfeld_update(points, weights, initial):
    new_facility = initial.copy()
    epsilon = 1e-6
    max_iter = 100

    for _ in range(max_iter):
        num = np.zeros_like(new_facility)
        denom = 0.0
        for i in range(len(points)):
            dist = np.linalg.norm(points[i] - new_facility)
            if dist < epsilon:
                continue
            w = weights[i] / dist
            num += w * points[i]
            denom += w
        if denom == 0:
            break
        updated = num / denom
        if np.linalg.norm(new_facility - updated) < epsilon:
            break
        new_facility = updated
    return new_facility

def generalized_weiszfeld_multi_facility_animated(points, weights, k, max_iter=100):
    points = np.array(points)
    weights = np.array(weights)

    rng = np.random.default_rng(seed=1)
    # Random initial facility locations not tied to demand points
    #facilities = rng.uniform(low=np.min(points, axis=0), high=np.max(points, axis=0), size=(k, 2))
    #facilities=np.array( [[6., 0.], [6., 0.], [6., 0.]])  # Example initial facilities for testing
    facilities=np.full((k, 2), [0., 10.])

    print("Initial facilities:\n", facilities)

    facility_history = [facilities.copy()]
    assignment_history = []

    for _ in range(max_iter):
        assignments = assign_points_to_facilities(points, facilities)
        assignment_history.append(assignments.copy())

        new_facilities = facilities.copy()
        for j in range(k):
            assigned_points = points[assignments == j]
            assigned_weights = weights[assignments == j]
            if len(assigned_points) > 0:
                new_facilities[j] = weiszfeld_update(assigned_points, assigned_weights, facilities[j])

        facility_history.append(new_facilities.copy())

        if np.linalg.norm(new_facilities - facilities) < 1e-6:
            break
        facilities = new_facilities

    return facility_history, assignment_history

def calculate_total_weighted_distance(points, weights, facilities, assignments):
    total_distance = 0.0
    for j in range(len(facilities)):
        assigned_points = points[assignments == j]
        assigned_weights = weights[assignments == j]
        for i in range(len(assigned_points)):
            dist = np.linalg.norm(assigned_points[i] - facilities[j])
            total_distance += assigned_weights[i] * dist
    return total_distance

def animate_facility_updates(points, weights, facility_history, assignment_history):
    k = facility_history[0].shape[0]
    colors = matplotlib.colormaps.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 7))
    plt.axis("off")
    fig.patch.set_facecolor("lightyellow")

    # Remove only the spines (border lines)
    for spine in ax.spines.values():
        spine.set_visible(False)


    def update(frame):
        ax.clear()
        ax.set_xlim(np.min(points[:, 0]) - 1, np.max(points[:, 0]) + 1)
        ax.set_ylim(np.min(points[:, 1]) - 1, np.max(points[:, 1]) + 1)
        #ax.set_title(f"Iteration {frame}")
        #ax.set_xlabel("X")
        #ax.set_ylabel("Y")
        ax.grid(False)
        #ax.set_visible(False)  # Hide the grid
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_facecolor("lightyellow")


        facilities = facility_history[frame]
        assignments = assignment_history[frame] if frame < len(assignment_history) else assignment_history[-1]

        for j in range(k):
            cluster_points = points[assignments == j]
            cluster_weights = weights[assignments == j]
            sizes = 20 + 2 * cluster_weights
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(((j+1)/ k)), s=sizes, label=f"Cluster {j+1}",edgecolors='black', alpha=0.7,marker='H')
            for point in cluster_points:
                ax.plot([point[0], facilities[j][0]], [point[1], facilities[j][1]],
                        color=colors(j / k), linestyle="--", linewidth=0.8)

        #for idx, (x, y) in enumerate(points):
        #    ax.text(x + 0.1, y + 0.1, f"{weights[idx]:.0f}", fontsize=8, color='black')

        for j in range(k):
            trace = np.array([fh[j] for fh in facility_history[:frame + 1]])
            ax.plot(trace[:, 0], trace[:, 1], color='red', linestyle='--', linewidth=1.5)

        ax.scatter(facilities[:, 0], facilities[:, 1], c="black", marker="X", s=150, label="Facilities")

        total_distance = calculate_total_weighted_distance(points, weights, facilities, assignments)
        ax.text(0.5, 0., f"Total Weighted Distance: {total_distance:.2f}", transform=ax.transAxes,
                ha='center', fontsize=12, color='blue')

        #ax.legend()
        ax.legend(loc='upper right', frameon=True, facecolor='lightyellow', edgecolor='none')

        return ax

    ani = FuncAnimation(fig, update, frames=len(facility_history), repeat=True, interval=600)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    rng = np.random.default_rng(seed=3)
    #demand_points = np.array([[1, 2], [2, 1], [3, 3], [8, 9], [9, 8],
    #                          [10, 10], [5, 5], [6, 6], [7, 7], [4, 4],
    #                          [3, 5], [8, 6], [6, 8], [2, 3], [9, 2],
    #                          [1, 8], [10, 5], [5, 10], [7, 9], [2, 6],
    #                          [3, 8], [4, 2], [6, 4], [8, 3], [9, 5],
    #                          [10, 7], [1, 9], [2, 4], [3, 6], [4, 8],
    #                          [5, 2], [6, 3], [7, 5], [8, 7], [9, 10],
    #                          [10, 1], [1, 3], [2, 5], [3, 7], [4, 9],
    #                          [5, 4], [6, 5], [7, 6], [8, 8], [9, 1],
    #                          [10, 2], [1, 4], [2, 7], [3, 9], [4, 10],
    #                          [5, 3], [6, 6], [7, 8], [8, 1], [9, 4],
    #                          [10, 3], [1, 5], [2, 8], [3, 10], [4, 1],
    #                          [5, 6], [6, 7], [7, 2], [8, 4], [9, 3],
    #                          [10, 8]])

    demand_points = rng.uniform(low=0, high=10, size=(150, 2))
    weights = rng.integers(1, 150, size=len(demand_points))  # Random integer weights
    k = 4  # Number of facilities

    facility_history, assignment_history = generalized_weiszfeld_multi_facility_animated(
        demand_points, weights, k
    )

    animate_facility_updates(demand_points, weights, facility_history, assignment_history)

    total_distance = calculate_total_weighted_distance(
        demand_points, weights, facility_history[-1], assignment_history[-1]
    )
    print(f"Total weighted distance: {total_distance:.2f}")
