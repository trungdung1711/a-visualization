import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

# - minority class has many points near boundary
# - minority also contains some "noisy" points inside the majority region
X, y = make_classification(
    n_classes=2,
    class_sep=0.5,  # small separation → overlap
    n_clusters_per_class=1,
    weights=[0.95, 0.05],  # imbalance
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_samples=500,
    random_state=105,
)

# 100, 11

# Manually inject noisy minority points deep in majority region
rng = np.random.RandomState(0)
n_noise = 8
noise_points = rng.normal(loc=[2.5, 2.5], scale=0.3, size=(n_noise, 2))
X = np.vstack([X, noise_points])
y = np.hstack([y, np.ones(n_noise)])  # mark them as minority

sm = SMOTE(k_neighbors=5, sampling_strategy=1.0)
X_sm, y_sm = sm.fit_resample(X, y)

bsm = BorderlineSMOTE(k_neighbors=5, sampling_strategy=1.0)
X_bsm, y_bsm = bsm.fit_resample(X, y)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))


def plot(ax, X, y, title):
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], alpha=0.5, label="Majority", s=20)
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], alpha=0.8, label="Minority", s=20)
    ax.set_title(title)
    ax.legend()


plot(axes[0], X, y, "Original Data (overlap + noisy minority)")
plot(axes[1], X_sm, y_sm, "Vanilla SMOTE (adds noise in majority region)")
plot(axes[2], X_bsm, y_bsm, "Borderline-SMOTE (focuses near boundary)")

plt.tight_layout()
plt.show()
