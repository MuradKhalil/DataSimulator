import numpy as np
from scipy import stats
from typing import List, Union, Tuple

class GaussianNormal:
    def __init__(self, distributions: List, n: int, random_state: int=None):
        """ Gaussian Normal copula to sample (non-normal) correlated data

        Args:
            distributions (List): list of probability distributions to convert marginal distributions to
            n (int): number of observations to generate
            random_state (int, optional): seed for reproducing results. Defaults to None.
        """
        self.distributions = distributions
        self.random_state = random_state
        self.n = n

    def _calculate_geometric_param(self, mu: Union[int, float]) -> Union[int, float]:
        """ Method to calculate the p parameter of the geometric distribution from the mean

        Args:
            mu (Union[int, float]): mean of the marginal distribution

        Returns:
            Union[int, float]: the p parameter of the geometric distribution
        """
        p = 1 / mu
        return p

    def _calculate_gamma_param(self, mu: Union[int, float], sigma: Union[int, float]) -> Tuple:
        """ Method to calculate the a and theta parameters of the gamma distribution from the mean and standard deviation

        Args:
            mu (Union[int, float]): mean of the marginal distribution
            sigma (Union[int, float]): standard deviation of the marginal distribution

        Returns:
            Tuple: a and theta parameter of the gamma distribution
        """
        theta = sigma**2 / mu
        a = mu / theta
        return a, theta

    def sample(self, cov: Union[List, np.array], mu: Union[List, np.array], sigma: Union[List, np.array]=None) -> np.array:
        """ Method to sample data from with specified probability distributions given a covariance/correlation matrix, 
        means and standard deviations of the marginal distributions 

        Args:
            cov (Union[List, np.array]): covariance or correlation matrix representing linear relationship between the marginal distributions
            mu (Union[List, np,array]): means of the marginal distributions
            sigma (Union[List, np.array], optional): standard deviations of the marginal distributions. Defaults to None.

        Raises:
            ValueError: if a probability distribution other than geometric, poisson, gamma, binomial or exponential is specified

        Returns:
            np.array: data sampled from the specified distributions
        """

        mu0 = np.zeros(len(mu))
        mv_dist = stats.multivariate_normal(mean=mu0, cov=cov)
        
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