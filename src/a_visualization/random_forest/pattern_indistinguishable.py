import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap


def plot_decision_boundary(X, y, model, ax, title):
    # Create a grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(["#AAAAFF", "#FFAAAA"])
    cmap_bold = ListedColormap(["#0000FF", "#FF0000"])

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k")
    ax.set_title(title)


# Case 1: Subtle but existing patterns
X1, y1 = make_classification(
    n_samples=500,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.5,
    flip_y=0.1,
    weights=[0.9, 0.1],
    random_state=212,
)
rf1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf1.fit(X1, y1)

# Case 2: Truly indistinguishable
X2, y2 = make_classification(
    n_samples=500,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.001,
    flip_y=0.0,
    weights=[0.9, 0.1],
    random_state=42,
)
rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf2.fit(X2, y2)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

plot_decision_boundary(X1, y1, rf1, axs[0], "Subtle Patterns (Separable)")
plot_decision_boundary(X2, y2, rf2, axs[1], "Truly Indistinguishable (Unseparable)")
plt.show()
