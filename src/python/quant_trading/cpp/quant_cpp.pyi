"""Type stubs for quant_cpp C++ extension module."""

from typing import List, Tuple
import numpy as np

__version__: str

class heston:
    """Heston stochastic volatility model submodule."""

    class OptionGreeks:
        """Option Greeks (sensitivities)."""
        delta: float
        gamma: float
        vega: float
        theta: float
        rho: float
        def __init__(self) -> None: ...

    class PricingResult:
        """Option pricing result with price and optional Greeks."""
        price: float
        greeks: "heston.OptionGreeks"
        greeks_computed: bool
        def __init__(self) -> None: ...

    class HestonParameters:
        """Heston model parameters."""
        kappa: float
        theta: float
        sigma: float
        rho: float
        v0: float

        def __init__(self) -> None: ...
        def __init__(
            self,
            kappa: float,
            theta: float,
            sigma: float,
            rho: float,
            v0: float,
        ) -> None: ...
        def is_feller_satisfied(self) -> bool: ...
        def is_valid(self) -> bool: ...
        def validate(self) -> None: ...

    class HestonModel:
        """Heston stochastic volatility model for option pricing."""

        def __init__(self, params: "heston.HestonParameters") -> None: ...
        def parameters(self) -> "heston.HestonParameters": ...
        def set_parameters(self, params: "heston.HestonParameters") -> None: ...
        def characteristic_function(
            self,
            u: complex,
            T: float,
            S0: float,
            r: float,
            q: float,
        ) -> complex: ...
        def price_option(
            self,
            strike: float,
            maturity: float,
            spot: float,
            rate: float,
            dividend: float,
            is_call: bool = True,
        ) -> float: ...
        def price_option_with_greeks(
            self,
            strike: float,
            maturity: float,
            spot: float,
            rate: float,
            dividend: float,
            is_call: bool = True,
        ) -> "heston.PricingResult": ...
        def price_options(
            self,
            strikes: List[float],
            maturities: List[float],
            spot: float,
            rate: float,
            dividend: float,
            is_call: bool = True,
        ) -> List[float]: ...
        def implied_volatility(
            self,
            strike: float,
            maturity: float,
            spot: float,
            rate: float,
            dividend: float,
            is_call: bool = True,
        ) -> float: ...


class sabr:
    """SABR volatility model submodule."""

    class SABRParameters:
        """SABR model parameters."""
        alpha: float
        beta: float
        rho: float
        nu: float

        def __init__(self) -> None: ...
        def __init__(
            self,
            alpha: float,
            beta: float,
            rho: float,
            nu: float,
        ) -> None: ...
        def is_valid(self) -> bool: ...
        def validate(self) -> None: ...

    class SABRModel:
        """SABR volatility model."""
        beta: float

        def __init__(self, beta: float = 0.5) -> None: ...
        def implied_volatility(
            self,
            strike: float,
            forward: float,
            maturity: float,
            alpha: float,
            rho: float,
            nu: float,
        ) -> float: ...
        def atm_volatility(
            self,
            forward: float,
            maturity: float,
            alpha: float,
            rho: float,
            nu: float,
        ) -> float: ...
        def implied_volatilities(
            self,
            strikes: List[float],
            forward: float,
            maturity: float,
            alpha: float,
            rho: float,
            nu: float,
        ) -> List[float]: ...
        def volatility_sensitivities(
            self,
            strike: float,
            forward: float,
            maturity: float,
            alpha: float,
            rho: float,
            nu: float,
        ) -> Tuple[float, float, float]: ...


class ou:
    """Ornstein-Uhlenbeck process submodule."""

    class OUParameters:
        """Ornstein-Uhlenbeck process parameters."""
        theta: float
        mu: float
        sigma: float

        def __init__(self) -> None: ...
        def __init__(
            self,
            theta: float,
            mu: float,
            sigma: float,
        ) -> None: ...
        def half_life(self) -> float: ...
        def is_mean_reverting(self) -> bool: ...
        def stationary_variance(self) -> float: ...
        def stationary_std(self) -> float: ...
        def is_valid(self) -> bool: ...
        def validate(self) -> None: ...

    class OUFitResult:
        """MLE fitting result for Ornstein-Uhlenbeck process."""
        params: "ou.OUParameters"
        log_likelihood: float
        aic: float
        bic: float
        n_observations: int
        converged: bool
        message: str

        def __init__(self) -> None: ...

    class OUProcess:
        """Ornstein-Uhlenbeck process utilities (static methods)."""

        @staticmethod
        def fit_mle(
            prices: List[float],
            dt: float = 1.0 / 252.0,
        ) -> "ou.OUFitResult": ...

        @staticmethod
        def log_likelihood(
            prices: List[float],
            params: "ou.OUParameters",
            dt: float = 1.0 / 252.0,
        ) -> float: ...

        @staticmethod
        def conditional_mean(
            x_t: float,
            params: "ou.OUParameters",
            dt: float,
        ) -> float: ...

        @staticmethod
        def conditional_variance(
            params: "ou.OUParameters",
            dt: float,
        ) -> float: ...

        @staticmethod
        def transition_density(
            x_next: float,
            x_t: float,
            params: "ou.OUParameters",
            dt: float,
        ) -> float: ...

        @staticmethod
        def simulate(
            params: "ou.OUParameters",
            x0: float,
            T: float,
            n_steps: int,
            seed: int = 42,
        ) -> List[float]: ...

        @staticmethod
        def optimal_boundaries(
            params: "ou.OUParameters",
            transaction_cost: float,
            risk_free_rate: float,
        ) -> Tuple[float, float, float]: ...
