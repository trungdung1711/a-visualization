import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.over_sampling import BorderlineSMOTE

# =========================================================
# 1. Create specially designed dataset
# =========================================================
# - Minority cluster close to majority cluster
# - Some majority intruding into minority region
# - Some minority right at the boundary

X, y = make_classification(
    n_classes=2,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.6,  # ensure overlap
    weights=[0.90, 0.10],
    random_state=17,
)

# add extra majority points inside minority region to create "hard boundary"
rng = np.random.RandomState(0)
intrusion = rng.normal(loc=[-1, 1], scale=0.25, size=(15, 2))
X = np.vstack([X, intrusion])
y = np.hstack([y, np.zeros(15)])  # label them as majority

# add some minority points extremely close to boundary
minority_boundary = rng.normal(loc=[0, 0], scale=0.2, size=(10, 2))
X = np.vstack([X, minority_boundary])
y = np.hstack([y, np.ones(10)])  # minority


# =========================================================
# 2. Apply Borderline-SMOTE1 and Borderline-SMOTE2
# =========================================================
bl1 = BorderlineSMOTE(kind="borderline-1", k_neighbors=5)
X_bl1, y_bl1 = bl1.fit_resample(X, y)

bl2 = BorderlineSMOTE(kind="borderline-2", k_neighbors=5)
X_bl2, y_bl2 = bl2.fit_resample(X, y)


# =========================================================
# 3. Visualization
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))


def plot(ax, X, y, title):
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=20, alpha=0.5, label="Majority")
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=25, alpha=0.8, label="Minority")
    ax.legend()
    ax.set_title(title)


plot(axes[0], X, y, "Original Overlapped Data\n(boundary + intrusion)")
plot(axes[1], X_bl1, y_bl1, "Borderline-SMOTE1\n(conservative: minority-minority)")
plot(axes[2], X_bl2, y_bl2, "Borderline-SMOTE2\n(aggressive: minority-majority)")

plt.tight_layout()
plt.show()
