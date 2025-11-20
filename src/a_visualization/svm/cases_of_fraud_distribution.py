import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# imbalanced data
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.96, 0.04],
    class_sep=0.4,
    random_state=17112004,
)

# 44

scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

# Without loss penalty
svm_default = SVC(kernel="linear")
# With loss penalty for the minority class
svm_balanced = SVC(kernel="linear", class_weight="balanced")

svm_default.fit(X_scale, y)
svm_balanced.fit(X_scale, y)


def plot_two_boundaries(model1, title1, model2, title2, X, y, x_range=5, y_range=5):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    global_xlim = (-x_range, x_range)
    global_ylim = (-y_range, y_range)

    models = [(model1, title1, axes[0]), (model2, title2, axes[1])]

    for model, title, ax in models:
        # Create a meshgrid for background coloring
        xx, yy = np.meshgrid(
            np.linspace(global_xlim[0], global_xlim[1], 500),
            np.linspace(global_ylim[0], global_ylim[1], 500),
        )
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision regions
        ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)

        # Scatter the actual data points
        ax.scatter(
            X[y == 0][:, 0], X[y == 0][:, 1], alpha=0.5, label="0", edgecolor="k"
        )
        ax.scatter(
            X[y == 1][:, 0], X[y == 1][:, 1], alpha=0.8, label="1", edgecolor="k"
        )

        # Plot the decision boundary
        w = model.coef_[0]
        b = model.intercept_[0]
        yy_boundary = (
            -(w[0] * np.linspace(global_xlim[0], global_xlim[1], 200) + b) / w[1]
        )
        ax.plot(
            np.linspace(global_xlim[0], global_xlim[1], 200),
            yy_boundary,
            linewidth=2,
            color="red",
        )

        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.show()


plot_two_boundaries(
    svm_default,
    "SVM Without Class Weights",
    svm_balanced,
    "SVM With Class Weights",
    X_scale,
    y,
    x_range=5,
    y_range=5,
)
