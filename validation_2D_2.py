import numpy as np
from numpy.random import multivariate_normal as mvn
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture as GMM_sk
from GMM_Conditional import *

"""
Verifying our code that finds the conditional distributions of Gaussian
Mixture Models. Specifically, we look at a 2D Gaussian Mixture Model,
with 2 components, and aim to evaluate the PDF p(xx | x1) for a fixed value
of x1 and a range of x2 values.

P.L.Green
"""

# Generate samples from a Gaussian Mixture Model
n_components=2
X1 = mvn(mean=[3, 3], cov=np.eye(2), size=50000)
X2 = mvn(mean=[-4, 0], cov=np.array([[1, 0.8],[0.8, 1]]), size=50000)
X = np.vstack([X1, X2])

# Fit a Gaussian mixture model
gmm = GMM_sk(n_components=n_components)
gmm.fit(X)

# Our conditional value of x1
x1 = -5

# Histogram approximation of conditional distribution
indices = np.where((X[:, 0] > x1 - 0.1) &
                   (X[:, 0] < x1 + 0.1))

# Create historgram
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(X[:, 0], X[:, 1], 'k x')
ax[0].plot(X[indices, 0], X[indices, 1], 'r o')
ax[0].set_xlabel('x1')
ax[0].set_ylabel('x2')
ax[0].set_ylim([-8, 8])
ax[1].hist(X[indices, 1].T, density=True, orientation=u'horizontal')
##ax[1].set_xlim([-8, 8])

# Create our GMM_Conditional object
gmm_cond = GMM_Conditional(means=gmm.means_,
                           covariances=gmm.covariances_,
                           weights=gmm.weights_,
                           n_components=gmm.n_components,
                           i_cond=np.array([True, False]))


# Plot conditional distribution over range of x1 values
x2_range = np.linspace(-7, 7, 1000)
pdf = np.zeros(1000)
for i in range(1000):
    pdf[i] = gmm_cond.pdf_xa_cond_xb(x2_range[i], x1)
ax[1].plot(pdf, x2_range, 'k')
ax[1].set_ylim([-8, 8])
ax[1].set_xlabel('p(x2 | x1)')
ax[1].set_ylabel('x2')

# Tidy-up and display plots
plt.tight_layout()
plt.show()
