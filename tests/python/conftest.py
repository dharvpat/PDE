"""
Pytest configuration for quant_trading tests.
"""

import pytest
import numpy as np


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n = 252  # One year of daily data
    returns = np.random.normal(0.0005, 0.02, n)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices


@pytest.fixture
def sample_option_data():
    """Generate sample option data for calibration tests."""
    return {
        "strikes": np.array([90, 95, 100, 105, 110]),
        "maturities": np.array([0.25, 0.5, 1.0]),
        "spot": 100.0,
        "rate": 0.05,
    }
