import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --------------------------
# Define two overlapping classes
# --------------------------
mu1, sigma1 = 0, 1  # Class 0 distribution
mu2, sigma2 = 1.5, 1  # Class 1 distribution

# X-axis range
x = np.linspace(-4, 6, 1000)

# Probability density functions
p0 = norm.pdf(x, mu1, sigma1)
p1 = norm.pdf(x, mu2, sigma2)

# --------------------------
# Bayes Optimal Decision Boundary
# p0(x) = p1(x)
# --------------------------
idx = np.argwhere(np.diff(np.sign(p0 - p1))).flatten()
decision_boundary = x[idx][0]

# --------------------------
# Compute Bayes Error
# Integral of min(p0, p1)
# --------------------------
bayes_error = np.trapezoid(np.minimum(p0, p1), x)
print("Bayes Error =", bayes_error)

# --------------------------
# Visualization
# --------------------------
plt.figure(figsize=(10, 5))

plt.plot(x, p0, label="Class 0 PDF", linewidth=2)
plt.plot(x, p1, label="Class 1 PDF", linewidth=2)

# Shade overlap (Bayes Error)
plt.fill_between(
    x, np.minimum(p0, p1), alpha=0.4, color="gray", label="Bayes Error (Overlap)"
)

# Mark decision boundary
plt.axvline(
    decision_boundary,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Bayes Boundary = {decision_boundary:.2f}",
)

plt.title("Visualization of Bayes Error (Overlap Between Class Distributions)")
plt.xlabel("Feature value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
