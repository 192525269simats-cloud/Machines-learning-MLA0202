
# Expectation Maximization Algorithm using GMM

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Create sample dataset
X = np.array([
    [1, 2], [2, 3], [3, 2], [2, 4],
    [7, 8], [8, 9], [9, 7], [8, 6],
    [4, 5], [5, 4], [6, 5], [5, 6]
])

# Create Gaussian Mixture Model
gmm = GaussianMixture(
    n_components=2,
    random_state=42
)

# Train the model using EM algorithm
gmm.fit(X)

# Predict cluster labels
labels = gmm.predict(X)

# Display results
print("Expectation Maximization Algorithm")
print("Number of Clusters:", gmm.n_components)
print("\nCluster Labels:")
print(labels)

print("\nCluster Means:")
print(gmm.means_)

print("\nModel Converged:", gmm.converged_)

# Plot clustering results
plt.scatter(
    X[:, 0], X[:, 1],
    c=labels,
    cmap="viridis",
    s=100
)

# Plot cluster centers
plt.scatter(
    gmm.means_[:, 0],
    gmm.means_[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Cluster Centers"
)

plt.title("Expectation Maximization Clustering")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.legend()
plt.grid(True)
plt.show()
