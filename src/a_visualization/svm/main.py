import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# imbalanced data
X, y = make_classification(
    n_samples=600,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.94, 0.06],
    class_sep=1.25,
    random_state=90,
)

scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

# Without loss penalty
svm_default = SVC(kernel="linear")
# With loss penalty for the minority class
svm_balanced = SVC(kernel="linear", class_weight="balanced")

svm_default.fit(X_scale, y)
svm_balanced.fit(X_scale, y)


def plot_two_boundaries(model1, title1, model2, title2, X, y):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    global_xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    global_ylim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

    models = [(model1, title1, axes[0]), (model2, title2, axes[1])]

    for model, title, ax in models:
        ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], alpha=0.5, label="0")
        ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], alpha=0.8, label="1")

        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)

        xx = np.linspace(global_xlim[0], global_xlim[1], 200)

        # decision boundary
        w = model.coef_[0]
        b = model.intercept_[0]
        yy = -(w[0] * xx + b) / w[1]

        ax.plot(xx, yy, linewidth=2)
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


plot_two_boundaries(
    svm_default,
    "SVM Without Class Weights",
    svm_balanced,
    "SVM With Class Weights",
    X_scale,
    y,
)
