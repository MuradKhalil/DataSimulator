import sys
sys.path.append('../src')
import pytest
from data_simulator.copula.gaussian_normal import GaussianNormal

@pytest.fixture
def gaussian_normal():
    return GaussianNormal(distributions=['gamma', 'geometric'], n=10)

@pytest.mark.parametrize('mu, expected',
    [
        (5, .2),
        (4, .25)
    ]
)
def test_gaussian_normal_calculate_geometric_param(gaussian_normal, mu, expected):
    assert gaussian_normal._calculate_geometric_param(mu) == expected

@pytest.mark.parametrize('mu, sigma, expected',
    [
        (5, 4, (1.5625, 3.2)),
        (4, 2, (4, 1))
    ]
)
def test_gaussian_normal_calculate_gamma_param(gaussian_normal, mu, sigma, expected):
    assert gaussian_normal._calculate_gamma_param(mu, sigma) == expected