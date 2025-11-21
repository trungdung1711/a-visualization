import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 37


def plot_decision_boundary(X, y, model, ax, title):
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.2, levels=np.arange(-0.5, 2), colors=["red", "blue"])
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=50, cmap=plt.cm.bwr)
    ax.set_title(title)


# Case 1: Clear separation, imbalance
X1, y1 = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],
    class_sep=2.0,
    random_state=RANDOM_STATE,
)

# Train linear SVM on original data
svm1 = SVC(kernel="linear", class_weight=None)
svm1.fit(X1, y1)

# Oversample to 1:1
ros = SMOTE(sampling_strategy=1.0, random_state=42)
X1_res, y1_res = ros.fit_resample(X1, y1)
svm1_res = SVC(kernel="linear")
svm1_res.fit(X1_res, y1_res)

# Case 2: Deeply overlapping minority inside majority
X2, y2 = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],
    class_sep=1.0,
    flip_y=0.1,
    random_state=RANDOM_STATE,
)

# train on original
svm2 = SVC(kernel="linear")
svm2.fit(X2, y2)

# Oversample to 1:1
ros_2 = SMOTE(sampling_strategy=1.0, random_state=42)
X2_res, y2_res = ros_2.fit_resample(X2, y2)
svm2_res = SVC(kernel="linear")
svm2_res.fit(X2_res, y2_res)


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 4 sub-plots
plot_decision_boundary(X1, y1, svm1, axes[0, 0], "Case 1: Original Imbalanced")
plot_decision_boundary(X1_res, y1_res, svm1_res, axes[0, 1], "Case 1: 1:1 Oversampling")

plot_decision_boundary(
    X2, y2, svm2, axes[1, 0], "Case 2: Original Imbalanced with Overlap"
)
plot_decision_boundary(
    X2_res, y2_res, svm2_res, axes[1, 1], "Case 2: 1:1 Oversampling (Noisy)"
)

plt.tight_layout()
plt.show()
