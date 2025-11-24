import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)

normal = np.random.randn(1000, 2) * 0.5

anomalies = np.random.uniform(low=3, high=6, size=(80, 2))

X = np.vstack([normal, anomalies])
y = np.array([0] * len(normal) + [1] * len(anomalies))  # 0 = normal, 1 = anomaly

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train autoencoder only on normal samples
X_train_norm = X_train[y_train == 0]
X_train_norm = torch.tensor(X_train_norm, dtype=torch.float32)


# Autoencoder's structure
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),  # bottleneck 1D latent space
        )

        self.decoder = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2))

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


for epoch in range(200):
    optimizer.zero_grad()
    # reconstruction point
    recon = model(X_train_norm)

    # output is connected with loss for backpropagation
    loss = criterion(recon, X_train_norm)
    loss.backward()
    optimizer.step()


print("Training finished. Final loss:", float(loss))

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
recon_test = model(X_test_tensor).detach().numpy()

mse = np.mean((recon_test - X_test) ** 2, axis=1)

plt.figure(figsize=(6, 5))
plt.scatter(X_test[:, 0], X_test[:, 1], c=mse, cmap="plasma")
plt.colorbar(label="Reconstruction Error")
plt.title("Autoencoder Reconstruction Error")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
