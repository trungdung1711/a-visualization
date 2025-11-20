import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

np.random.seed(42)

# # Data: normal ring + scattered anomalies
# # random angles from 0 to 2pi ~800
# angles = np.random.uniform(0, 2 * np.pi, 800)
# # radius
# r = 1 + 0.1 * np.random.randn(800)
# # the rings
# normal = np.column_stack((r * np.cos(angles), r * np.sin(angles)))

# # anomalies inside
# anomalies = np.random.uniform(-1.2, 1.2, (50, 2))

# X = np.vstack([normal, anomalies])


# # Normal cluster + anomalies far away
# normal = np.random.randn(800, 2)  # centered at (0,0)

# # anomalies far away
# anomalies = np.random.uniform(low=-6, high=6, size=(50, 2))

# X = np.vstack([normal, anomalies])

# 2 Normal blobs with random anomalies

normal, _ = make_blobs(
    n_samples=800, centers=[[-2, 0], [2, 0]], cluster_std=0.5, random_state=42
)
anomalies = np.random.uniform(low=-6, high=6, size=(50, 2))

X = np.vstack([normal, anomalies])


iso = IsolationForest(contamination=0.06, random_state=42)

iso.fit(X)

# range [-1, 1], more posive -> normals, more negative -> anomalies
scores = iso.decision_function(X)
preds = iso.predict(X)

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=scores, cmap="Oranges")
plt.title("Isolation Forest Anomaly Detection\n(Color = Normality Score)")
plt.colorbar(label="Normality Score")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect("equal", "box")
plt.show()
