import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal as Normal_PDF

"""
Takes the parameters of a Gaussian Mixture Model as an input, and creates
an object where we can evaluate the GMM's conditional distributions.

To do:

1. So far only outputs p(x1 | x2), but implementing p(x2 | x1) should be
trivial

2. Only tested on 2D Gaussian Mixture Models so far, but SHOULD be OK on
higher dimensional problems.

3. Currently only takes scale values of x1 into 'pdf_x1_cond_x2'. Should be
easy to generalise though.

P.L.Green
"""


class GMM_Conditional:

    def __init__(self, means, covariances, weights, n_components, D1, D2):
        """ Initialiser class method.

            means - means of Gaussians in mixture
            covariances - covariance matrices of Gaussians in mixture
            weights - mixture proportions of the GMM
            n_components - no. of components in the GMM
            D1 - dimension of x1 (see notes)
            D2 - dimension of x2 (see notes)

        """

        self.n_components = n_components
        self.weights = weights

        # Initialise lists
        self.mu_11_list = []
        self.mu_22_list = []
        self.Sigma_11_list = []
        self.Sigma_12_list = []
        self.Sigma_21_list = []
        self.Sigma_22_list = []

        # Isolate components of Gaussian Mixture model
        for c in range(n_components):

            # Split mean into individual components
            mu_11, mu_22 = means[c][0:D1], means[c][D1:D1 + D2]

            # Split covariance matrix into individual components
            (Sigma_11,
             Sigma_12,
             Sigma_21,
             Sigma_22) = (covariances[c][0:D1, 0:D1],
                          covariances[c][0:D1, D1:D1 + D2],
                          covariances[c][D1:D1 + D2, 0:D1],
                          covariances[c][D1:D1 + D2, D1:D1 + D2])

            self.mu_11_list.append(mu_11)
            self.mu_22_list.append(mu_22)
            self.Sigma_11_list.append(Sigma_11)
            self.Sigma_12_list.append(Sigma_12)
            self.Sigma_21_list.append(Sigma_21)
            self.Sigma_22_list.append(Sigma_22)

        # Create lists of marginal probability distributions
        self.p_11 = []
        self.p_22 = []
        for c in range(n_components):
            self.p_11.append(Normal_PDF(mean=self.mu_11_list[c],
                                        cov=self.Sigma_11_list[c]))
            self.p_22.append(Normal_PDF(mean=self.mu_22_list[c],
                                        cov=self.Sigma_22_list[c]))

    def mu_1_2(self, x2, mu_11, Sigma_12, Sigma_22, mu_22):
        """ Expression for the mean of each Gaussian that makes up p(x1 | x2)

        """

        mu = mu_11 + Sigma_12 @ inv(Sigma_22) @ (x2 - mu_22)

        return mu

    def Sigma_1_2(self, x2, Sigma_11, Sigma_12, Sigma_21, Sigma_22):
        """ Expression for the covariance matrix of each Gaussian
            that makes up p(x1 | x2)

        """

        Sigma = Sigma_11 - Sigma_12 @ inv(Sigma_22) @ Sigma_21

        return Sigma

    def w_1_2(self, x2):
        """ Mixture weights, conditional on x2

        """

        # Array of components that will make up the denominator of our
        # conditional weights expression.
        den_components = np.zeros(self.n_components)
        for c in range(self.n_components):
            den_components[c] = self.p_22[c].pdf(x2) * self.weights[c]

        # Find denominator
        den = np.sum(den_components)

        # Initialise list of new, conditional weights
        new_weights = []

        # Find new weights
        for c in range(self.n_components):
            new_weights.append(den_components[c] / den)

        return new_weights

    def pdf_x1_cond_x2(self, x1, x2):
        """ Compute the probability density of x1, given x2.

        """

        # Find Gaussian components of p(x1 | x2)
        p_1_2 = []
        for c in range(self.n_components):

            mu = self.mu_1_2(x2, self.mu_11_list[c], self.Sigma_12_list[c],
                             self.Sigma_22_list[c], self.mu_22_list[c])

            Sigma = self.Sigma_1_2(x2, self.Sigma_11_list[c],
                                   self.Sigma_12_list[c],
                                   self.Sigma_21_list[c],
                                   self.Sigma_22_list[c])

            p_1_2.append(Normal_PDF(mean=mu, cov=Sigma))

        # Find weights of p(x1 | x2)
        weights_1_2 = self.w_1_2(x2)

        # Calculate pdf, p(x1 | x2)
        pdf = np.zeros(1)
        for c in range(self.n_components):
            pdf += weights_1_2[c] * p_1_2[c].pdf(x1)

        return pdf
