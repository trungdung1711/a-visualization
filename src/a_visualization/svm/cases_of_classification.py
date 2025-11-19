import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def plot_svm_subplot(ax, X, y, title):
    clf = SVC(kernel="linear", C=1.0)
    clf.fit(X, y)

    # plot points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=50, edgecolors="k")

    # set limits based on data
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # create grid
    x_sam = np.linspace(x_min, x_max, 500)
    y_sam = np.linspace(y_min, y_max, 500)
    xx, yy = np.meshgrid(x_sam, y_sam)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot decision boundary and regions
    ax.contour(xx, yy, Z, levels=[0], colors="k", linestyles="--")
    ax.contourf(xx, yy, Z > 0, alpha=0.1, colors=["red", "blue"])

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


# Generate data
np.random.seed(42)
legit = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)

# Scenario 1: fraud near boundary
fraud1 = np.random.multivariate_normal([1.5, 1.5], [[1, 0], [0, 1]], 20)
X1 = np.vstack([legit, fraud1])
y1 = np.array([0] * 100 + [1] * 20)

# Scenario 2: fraud deep inside legitimate region
fraud2 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 20)
X2 = np.vstack([legit, fraud2])
y2 = np.array([0] * 100 + [1] * 20)

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_svm_subplot(axes[0], X1, y1, "Scenario 1: Can be separated")
plot_svm_subplot(axes[1], X2, y2, "Scenario 2: Deeply messy")
plt.tight_layout()
plt.show()
