import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as Normal_PDF
from sklearn.mixture import GaussianMixture as GMM_sk
from GMM_Conditional import *

"""
Verifying our code that finds the conditional distributions of Gaussian
Mixture Models. Specifically, we look at a 2D Gaussian Mixture Model,
with 2 components, and aim to evaluate the PDF p(x1 | x2) for a fixed value
of x2 and a range of x1 values.

To-do list is written in the GMM_Conditional.py class.

P.L.Green
"""

# Load 2D data from a mixture of 2 Gaussians
file = open('2_cluster_data_unsupervised.dat', 'rb')
X, N = pickle.load(file)
n_components = 2
file.close()

# Fit a Gaussian mixture model
gmm = GMM_sk(n_components)
gmm.fit(X)

# Initialise plots
fig, ax = plt.subplots(nrows=2, ncols=1)

# Plot data
ax[0].plot(X[:, 0], X[:, 1], 'k o', alpha=0.3)

# Plot contours of GMM
r1 = np.linspace(-7, 7, 100)
r2 = np.linspace(-7, 7, 100)
x_r1, x_r2 = np.meshgrid(r1, r2)
pos = np.empty(x_r1.shape + (2, ))
pos[:, :, 0] = x_r1
pos[:, :, 1] = x_r2
for c in range(n_components):
    p = Normal_PDF(gmm.means_[c], gmm.covariances_[c])
    ax[0].contour(x_r1, x_r2, gmm.weights_[c] * p.pdf(pos))

# Create our GMM_Conditional object
x2 = 1      # Our conditional value of x2
gmm_cond = GMM_Conditional(means=gmm.means_,
                           covariances=gmm.covariances_,
                           weights=gmm.weights_,
                           n_components=gmm.n_components,
                           D1=1, D2=1)

# Plot line showing conditional value of x2
ax[0].plot([-7, 7], np.repeat(x2, 2), 'r')
ax[0].set_xlim([-7, 7])
ax[0].set_ylim([-7, 7])
ax[0].set_xlabel('$x_1$')
ax[0].set_ylabel('$x_2$')

# Plot conditional distribution over range of x1 values
x1_range = np.linspace(-7, 7, 1000)
pdf = np.zeros(1000)
for i in range(1000):
    pdf[i] = gmm_cond.pdf_x1_cond_x2(x1_range[i], x2)
ax[1].plot(x1_range, pdf, 'k')
ax[1].set_xlim([-7, 7])
ax[1].set_xlabel('x1')
ax[1].set_ylabel('p(x1 | x2)')

# Tidy-up and display plots
plt.tight_layout()
plt.show()
