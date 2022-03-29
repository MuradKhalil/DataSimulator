import numpy as np
from scipy import stats
from typing import List

class DataSimulation:
    def __init__(self, distributions: List, n: int, random_state: int=None, copula: str='Gaussian Normal'):
        self.copula = copula
        self.distributions = distributions
        self.random_state = random_state
        self.n = n

    def _calculate_geometric_param(self, mu):
        p = 1 / mu
        return p

    def _calculate_gamma_param(self, mu, sigma):
        theta = sigma**2 / mu
        a = mu / theta
        return a, theta

    def simulate(self, cov, mu, sigma=None):
        """
        Simulate correlated data based on copula. 
        Either a correlation or covariance matrix can be passed here
        """

        mu0 = np.zeros(len(mu))
        if self.copula == 'Gaussian Normal':
            mv_dist = stats.multivariate_normal(mean=mu0, cov=cov)
        else:
            raise ValueError('Only Gaussian Normal supported currently')
        
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = mv_dist.rvs(self.n)
        norm = stats.norm()
        X_unif = norm.cdf(X)

        feats = []

        for i, dist in enumerate(self.distributions):
            if dist == 'geometric':
                geometric = stats.geom(p=self._calculate_geometric_param(mu[i]))
                feat = geometric.ppf(X_unif[:, i]).reshape((self.n, 1))

            elif dist == 'poisson':
                poisson = stats.poisson(mu=mu[i])
                feat = poisson.ppf(X_unif[:, i]).reshape((self.n, 1))

            elif dist == 'gamma':
                a, theta = self._calculate_gamma_param(mu[i], sigma[i])
                gamma = stats.gamma(a=a, scale=theta)
                feat = gamma.ppf(X_unif[:, i]).reshape((self.n, 1))

            elif dist == 'binomial':
                binomial = stats.binom(n=1, p=mu[i])
                feat = binomial.ppf(X_unif[:, i]).reshape((self.n, 1))

            elif dist == 'exponential':
                exponential = stats.expon(scale=mu[i])
                feat = exponential.ppf(X_unif[:, i]).reshape((self.n, 1))

            else:
                raise ValueError(f'{dist} distribution is not supported currently')

            feats.append(feat)

        return np.concatenate(feats, axis=1)