import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal as Normal_PDF

"""
Takes the parameters of a Gaussian Mixture Model as an input, and creates
an object where we can evaluate the GMM's conditional distributions.

P.L.Green
"""


class GMM_Conditional:

    def __init__(self, means, covariances, weights, n_components,
                 i_cond):
        """ Initialiser class method.

            means - means of Gaussians in mixture

            covariances - covariance matrices of Gaussians in mixture

            weights - mixture proportions of the GMM

            n_components - no. of components in the GMM

            i_cond - True values are the locations of xb

        """

        self.n_components = n_components
        self.weights = weights
        self.D = len(means)
        self.i_cond = i_cond

        # Initialise lists
        self.mu_aa_list = []
        self.mu_bb_list = []
        self.Sigma_aa_list = []
        self.Sigma_ab_list = []
        self.Sigma_ba_list = []
        self.Sigma_bb_list = []

        # Isolate components of Gaussian Mixture model
        for c in range(n_components):

            # Split mean into individual components
            mu_aa, mu_bb = means[c][~i_cond], means[c][i_cond]

            # Split covariance matrix into individual components
            (Sigma_aa,
             Sigma_ab,
             Sigma_ba,
             Sigma_bb) = (covariances[c][~i_cond, ~i_cond],
                          covariances[c][~i_cond, i_cond],
                          covariances[c][i_cond, ~i_cond],
                          covariances[c][i_cond, i_cond])

            self.mu_aa_list.append(mu_aa)
            self.mu_bb_list.append(mu_bb)
            self.Sigma_aa_list.append(Sigma_aa)
            self.Sigma_ab_list.append(Sigma_ab)
            self.Sigma_ba_list.append(Sigma_ba)
            self.Sigma_bb_list.append(Sigma_bb)

        # Create lists of marginal probability distributions
        self.p_aa = []
        self.p_bb = []
        for c in range(n_components):
            self.p_aa.append(Normal_PDF(mean=self.mu_aa_list[c],
                                        cov=self.Sigma_aa_list[c]))
            self.p_bb.append(Normal_PDF(mean=self.mu_bb_list[c],
                                        cov=self.Sigma_bb_list[c]))

    def mu_a_b(self, xb, mu_aa, Sigma_ab, Sigma_bb, mu_bb):
        """ Expression for the mean of each Gaussian that makes up p(xa | xb)

        """

        mu = mu_aa + Sigma_ab @ inv(Sigma_bb) @ (xb - mu_bb)

        return mu

    def Sigma_a_b(self, xb, Sigma_aa, Sigma_ab, Sigma_ba, Sigma_bb):
        """ Expression for the covariance matrix of each Gaussian
            that makes up p(xa | xb)

        """

        Sigma = Sigma_aa - Sigma_ab @ inv(Sigma_bb) @ Sigma_ba

        return Sigma

    def w_a_b(self, xb):
        """ Mixture weights, conditional on xb

        """

        # Array of components that will make up the denominator of our
        # conditional weights expression.
        den_components = np.zeros(self.n_components)
        for c in range(self.n_components):
            den_components[c] = self.p_bb[c].pdf(xb) * self.weights[c]

        # Find denominator
        den = np.sum(den_components)

        # Initialise list of new, conditional weights
        new_weights = []

        # Find new weights
        for c in range(self.n_components):
            new_weights.append(den_components[c] / den)

        return new_weights

    def pdf_xa_cond_xb(self, xa, xb):
        """ Compute the probability density of xa, given xb.

        """

        # Find Gaussian components of p(xa | xb)
        p_a_b = []
        for c in range(self.n_components):

            mu = self.mu_a_b(xb, self.mu_aa_list[c], self.Sigma_ab_list[c],
                             self.Sigma_bb_list[c], self.mu_bb_list[c])

            Sigma = self.Sigma_a_b(xb, self.Sigma_aa_list[c],
                                   self.Sigma_ab_list[c],
                                   self.Sigma_ba_list[c],
                                   self.Sigma_bb_list[c])

            p_a_b.append(Normal_PDF(mean=mu, cov=Sigma))

        # Find weights of p(xa | xb)
        weights_a_b = self.w_a_b(xb)

        # Calculate pdf, p(xa | xb)
        pdf = np.zeros(1)
        for c in range(self.n_components):
            pdf += weights_a_b[c] * p_a_b[c].pdf(xa)

        return pdf
