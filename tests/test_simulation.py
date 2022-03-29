import sys
sys.path.append('../src')

import pytest
from data_simulator.copula.simulation import DataSimulation

import numpy as np

@pytest.fixture
def data_simulation_gumbel():
    return DataSimulation(copula='Gumbel', distributions=['gamma', 'geometric'], n=10)

def test_data_simulator_init(data_simulation_gumbel):
    with pytest.raises(ValueError):
        data_simulation_gumbel.simulate(cov=[[1, .5], [.5, 1]], mu=[0, 0], sigma=[0, 0])

@pytest.fixture
def data_simulation_normal():
    return DataSimulation(copula='Gaussian Normal', distributions=['gamma', 'geometric'], n=10)

@pytest.mark.parametrize('mu, expected',
    [
        (5, .2),
        (4, .25)
    ]
)
def test_data_simulation_calculate_geometric_param(data_simulation_normal, mu, expected):
    assert data_simulation_normal._calculate_geometric_param(mu) == expected

@pytest.mark.parametrize('mu, sigma, expected',
    [
        (5, 4, (1.5625, 3.2)),
        (4, 2, (4, 1))
    ]
)
def test_data_simulation_calculate_gamma_param(data_simulation_normal, mu, sigma, expected):
    assert data_simulation_normal._calculate_gamma_param(mu, sigma) == expected