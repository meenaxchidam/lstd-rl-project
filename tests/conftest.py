"""Shared mock data (plan Sec. 5): deterministic chain 2 -> 1 -> 0 (terminal), reward -1, gamma = 1."""
import numpy as np
import pytest


@pytest.fixture
def mock_phi():
    return np.eye(3)


@pytest.fixture
def mock_episodes():
    return [[(2, -1.0, 1, False), (1, -1.0, 0, True)]] * 50


@pytest.fixture
def mock_v_true():
    return np.array([0.0, -1.0, -2.0])
