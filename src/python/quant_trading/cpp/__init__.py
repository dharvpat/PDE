"""
C++ extension modules for high-performance quantitative finance.

This package provides Python bindings to C++ implementations of:
- Heston stochastic volatility model
- SABR volatility model
- Ornstein-Uhlenbeck mean-reverting process

The C++ implementations provide 10-100x speedups compared to pure Python
for computationally intensive operations.

Example:
    >>> from quant_trading.cpp import quant_cpp
    >>> # Access Heston model
    >>> params = quant_cpp.heston.HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
    >>> model = quant_cpp.heston.HestonModel(params)
    >>> price = model.price_option(100.0, 1.0, 100.0, 0.05, 0.02, True)
"""

try:
    from . import quant_cpp
    __all__ = ["quant_cpp"]
    _cpp_available = True
except ImportError as e:
    _cpp_available = False
    _import_error = str(e)
    __all__ = []


def is_available() -> bool:
    """Check if C++ extensions are available."""
    return _cpp_available


def get_import_error() -> str:
    """Get the import error message if C++ extensions are not available."""
    if _cpp_available:
        return ""
    return _import_error
