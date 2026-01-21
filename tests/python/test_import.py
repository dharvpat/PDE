"""
Basic import tests for quant_trading package.
"""

import pytest


def test_import_package():
    """Test that the main package can be imported."""
    import quant_trading

    assert quant_trading is not None


def test_version():
    """Test that version is defined and valid."""
    import quant_trading

    assert quant_trading.__version__ == "1.0.0"


def test_import_submodules():
    """Test that all submodules can be imported."""
    from quant_trading import calibration
    from quant_trading import models
    from quant_trading import signals
    from quant_trading import risk
    from quant_trading import execution

    assert calibration is not None
    assert models is not None
    assert signals is not None
    assert risk is not None
    assert execution is not None
