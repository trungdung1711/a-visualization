import numpy as np
import matplotlib.pyplot as plt


# Object function to minimize
def f(x, y):
    return x**2 + y**2 + 1


# flies number
n_flies = 5
# number of iteration
n_iter = 60
# initial search range
search_range = 70
# explore range to place the flies
explore_range = 1.5

# Initial random
positions = np.random.uniform(-search_range, search_range, (n_flies, 2))

trajectories = [positions.copy()]

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-search_range, search_range)
ax.set_ylim(-search_range, search_range)
ax.set_title("Fruit Fly Optimization Algorithm (2D) Swarm Convergence")
ax.scatter(0, 0, c="red", s=100, label="Global Optimum")
colors = plt.cm.jet(np.linspace(0, 1, n_flies))

scatters = [
    ax.scatter(positions[i, 0], positions[i, 1], color=colors[i], s=50)
    for i in range(n_flies)
]

lines = [ax.plot([], [], colors[i], alpha=0.6)[0] for i in range(n_flies)]

for t in range(n_iter):
    # Calculate the fitness for all flies
    fitness = np.array([f(x, y) for x, y in positions])

    # Choose the best flies
    best_idx = np.argmin(fitness)
    best_pos = positions[best_idx]

    # Moves the swarm
    positions += 0.5 * (best_pos - positions) + np.random.uniform(
        -explore_range, explore_range, positions.shape
    )

    trajectories.append(positions.copy())

    for i in range(n_flies):
        scatters[i].set_offsets(positions[i])
        traj = np.array([traj[i] for traj in trajectories])
        lines[i].set_data(traj[:, 0], traj[:, 1])

    ax.set_title(f"Iteration {t+1}, Best fitness: {fitness[best_idx]:.3f}")
    plt.pause(0.3)

plt.ioff()
plt.legend()
plt.show()
