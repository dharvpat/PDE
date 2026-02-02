"""
Unit tests for C++ Python bindings.

Tests the pybind11 bindings for Heston, SABR, and OU process models.
"""

import math
import pytest
import numpy as np


# Skip all tests if C++ bindings are not available
try:
    from quant_trading.cpp import quant_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CPP_AVAILABLE,
    reason="C++ bindings not available"
)


class TestHestonBindings:
    """Tests for Heston model C++ bindings."""

    def test_parameters_default_construction(self):
        """Test default parameter construction."""
        params = quant_cpp.heston.HestonParameters()
        assert params.kappa == 2.0
        assert params.theta == 0.04
        assert params.sigma == 0.3
        assert params.rho == -0.7
        assert params.v0 == 0.04

    def test_parameters_custom_construction(self):
        """Test custom parameter construction."""
        params = quant_cpp.heston.HestonParameters(3.0, 0.05, 0.4, -0.5, 0.06)
        assert params.kappa == 3.0
        assert params.theta == 0.05
        assert params.sigma == 0.4
        assert params.rho == -0.5
        assert params.v0 == 0.06

    def test_feller_condition(self):
        """Test Feller condition checking."""
        # Feller satisfied: 2 * 2.0 * 0.04 = 0.16 >= 0.3^2 = 0.09
        params = quant_cpp.heston.HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
        assert params.is_feller_satisfied()

        # Feller violated: 2 * 1.0 * 0.02 = 0.04 < 0.5^2 = 0.25
        params_bad = quant_cpp.heston.HestonParameters(1.0, 0.02, 0.5, -0.7, 0.04)
        assert not params_bad.is_feller_satisfied()

    def test_model_construction(self):
        """Test model construction."""
        params = quant_cpp.heston.HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
        model = quant_cpp.heston.HestonModel(params)
        assert model is not None

    def test_model_invalid_params(self):
        """Test model rejects invalid parameters."""
        params = quant_cpp.heston.HestonParameters(-1.0, 0.04, 0.3, -0.7, 0.04)
        with pytest.raises(ValueError):
            quant_cpp.heston.HestonModel(params)

    def test_price_call_option(self):
        """Test call option pricing."""
        params = quant_cpp.heston.HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
        model = quant_cpp.heston.HestonModel(params)

        price = model.price_option(
            strike=100.0,
            maturity=1.0,
            spot=100.0,
            rate=0.05,
            dividend=0.02,
            is_call=True
        )

        assert price > 0
        assert price < 100  # Less than spot
        assert 3.0 < price < 20.0  # Reasonable range for ATM call

    def test_price_put_option(self):
        """Test put option pricing."""
        params = quant_cpp.heston.HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
        model = quant_cpp.heston.HestonModel(params)

        price = model.price_option(
            strike=100.0,
            maturity=1.0,
            spot=100.0,
            rate=0.05,
            dividend=0.02,
            is_call=False
        )

        assert price > 0
        assert price < 100  # Less than strike

    def test_put_call_parity(self):
        """Test put-call parity holds."""
        params = quant_cpp.heston.HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
        model = quant_cpp.heston.HestonModel(params)

        S0, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02

        call = model.price_option(K, T, S0, r, q, is_call=True)
        put = model.price_option(K, T, S0, r, q, is_call=False)

        # Put-Call Parity: C - P = S*exp(-qT) - K*exp(-rT)
        expected_diff = S0 * math.exp(-q * T) - K * math.exp(-r * T)
        actual_diff = call - put

        assert abs(actual_diff - expected_diff) < 0.5

    def test_price_multiple_options(self):
        """Test vectorized option pricing."""
        params = quant_cpp.heston.HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
        model = quant_cpp.heston.HestonModel(params)

        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [1.0]

        prices = model.price_options(strikes, maturities, 100.0, 0.05, 0.02, True)

        assert len(prices) == len(strikes)
        assert all(p > 0 for p in prices)
        # Call prices should decrease with strike
        for i in range(1, len(prices)):
            assert prices[i] < prices[i - 1]

    def test_implied_volatility(self):
        """Test implied volatility calculation."""
        params = quant_cpp.heston.HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
        model = quant_cpp.heston.HestonModel(params)

        iv = model.implied_volatility(100.0, 1.0, 100.0, 0.05, 0.02, True)

        assert iv > 0.05
        assert iv < 1.0
        # Should be close to sqrt(v0) for ATM
        assert abs(iv - math.sqrt(0.04)) < 0.1

    def test_price_with_greeks(self):
        """Test pricing with Greeks."""
        params = quant_cpp.heston.HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
        model = quant_cpp.heston.HestonModel(params)

        result = model.price_option_with_greeks(100.0, 1.0, 100.0, 0.05, 0.02, True)

        assert result.price > 0
        assert result.greeks_computed

        # Call delta should be between 0 and 1
        assert 0 < result.greeks.delta < 1
        # ATM call delta should be around 0.5
        assert 0.3 < result.greeks.delta < 0.7

        # Gamma should be positive
        assert result.greeks.gamma > 0


class TestSABRBindings:
    """Tests for SABR model C++ bindings."""

    def test_parameters_default_construction(self):
        """Test default parameter construction."""
        params = quant_cpp.sabr.SABRParameters()
        assert params.alpha == 0.2
        assert params.beta == 0.5
        assert params.rho == -0.3
        assert params.nu == 0.4

    def test_model_construction(self):
        """Test model construction."""
        model = quant_cpp.sabr.SABRModel(beta=0.5)
        assert model.beta == 0.5

    def test_implied_volatility_atm(self):
        """Test ATM implied volatility."""
        model = quant_cpp.sabr.SABRModel(beta=0.5)

        vol = model.implied_volatility(
            strike=100.0,
            forward=100.0,
            maturity=1.0,
            alpha=0.2,
            rho=-0.3,
            nu=0.4
        )

        assert vol > 0
        assert vol < 2.0

    def test_atm_volatility(self):
        """Test ATM volatility formula."""
        model = quant_cpp.sabr.SABRModel(beta=0.5)

        vol = model.atm_volatility(
            forward=100.0,
            maturity=1.0,
            alpha=0.2,
            rho=-0.3,
            nu=0.4
        )

        assert vol > 0
        assert vol < 2.0

    def test_implied_volatilities_vectorized(self):
        """Test vectorized implied volatility calculation."""
        model = quant_cpp.sabr.SABRModel(beta=0.5)

        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        vols = model.implied_volatilities(
            strikes, 100.0, 1.0, 0.2, -0.3, 0.4
        )

        assert len(vols) == len(strikes)
        assert all(v > 0 for v in vols)
        assert all(v < 2.0 for v in vols)

    def test_volatility_sensitivities(self):
        """Test volatility sensitivities."""
        model = quant_cpp.sabr.SABRModel(beta=0.5)

        d_alpha, d_rho, d_nu = model.volatility_sensitivities(
            strike=100.0,
            forward=100.0,
            maturity=1.0,
            alpha=0.2,
            rho=-0.3,
            nu=0.4
        )

        # Sensitivities should be finite
        assert math.isfinite(d_alpha)
        assert math.isfinite(d_rho)
        assert math.isfinite(d_nu)


class TestOUBindings:
    """Tests for Ornstein-Uhlenbeck process C++ bindings."""

    def test_parameters_default_construction(self):
        """Test default parameter construction."""
        params = quant_cpp.ou.OUParameters()
        assert params.theta == 0.0
        assert params.mu == 1.0
        assert params.sigma == 0.1

    def test_parameters_half_life(self):
        """Test half-life calculation."""
        params = quant_cpp.ou.OUParameters(0.0, 5.0, 0.1)
        expected_half_life = math.log(2) / 5.0
        assert abs(params.half_life() - expected_half_life) < 1e-10

    def test_parameters_stationary_variance(self):
        """Test stationary variance calculation."""
        params = quant_cpp.ou.OUParameters(0.0, 5.0, 0.1)
        # Var = sigma^2 / (2 * mu) = 0.01 / 10 = 0.001
        assert abs(params.stationary_variance() - 0.001) < 1e-10

    def test_simulate_basic(self):
        """Test basic simulation."""
        params = quant_cpp.ou.OUParameters(0.0, 5.0, 0.1)
        path = quant_cpp.ou.OUProcess.simulate(params, 0.5, 1.0, 252, 42)

        assert len(path) == 253  # n_steps + 1
        assert path[0] == 0.5  # Initial value
        assert all(math.isfinite(x) for x in path)

    def test_simulate_reproducibility(self):
        """Test simulation reproducibility with same seed."""
        params = quant_cpp.ou.OUParameters(0.0, 5.0, 0.1)
        path1 = quant_cpp.ou.OUProcess.simulate(params, 0.0, 1.0, 100, 12345)
        path2 = quant_cpp.ou.OUProcess.simulate(params, 0.0, 1.0, 100, 12345)

        assert path1 == path2

    def test_fit_mle_basic(self):
        """Test basic MLE fitting."""
        params = quant_cpp.ou.OUParameters(0.0, 5.0, 0.1)
        path = quant_cpp.ou.OUProcess.simulate(params, 0.0, 1.0, 252, 42)

        result = quant_cpp.ou.OUProcess.fit_mle(path, 1.0 / 252.0)

        assert result.converged
        assert result.n_observations == len(path)
        assert math.isfinite(result.log_likelihood)

    def test_fit_mle_recovery(self):
        """Test MLE parameter recovery."""
        true_params = quant_cpp.ou.OUParameters(0.0, 5.0, 0.1)
        path = quant_cpp.ou.OUProcess.simulate(true_params, 0.0, 10.0, 2500, 42)

        result = quant_cpp.ou.OUProcess.fit_mle(path, 10.0 / 2500.0)

        assert result.converged
        # Check parameter recovery with generous tolerance
        assert abs(result.params.theta - true_params.theta) < 0.2
        assert abs(result.params.mu - true_params.mu) < 2.0
        assert abs(result.params.sigma - true_params.sigma) < 0.05

    def test_log_likelihood(self):
        """Test log-likelihood computation."""
        params = quant_cpp.ou.OUParameters(0.0, 5.0, 0.1)
        path = quant_cpp.ou.OUProcess.simulate(params, 0.0, 1.0, 252, 42)

        ll = quant_cpp.ou.OUProcess.log_likelihood(path, params, 1.0 / 252.0)

        assert math.isfinite(ll)
        assert not math.isnan(ll)

    def test_conditional_mean_variance(self):
        """Test conditional mean and variance."""
        params = quant_cpp.ou.OUParameters(0.0, 5.0, 0.1)
        dt = 1.0 / 252.0

        mean = quant_cpp.ou.OUProcess.conditional_mean(0.5, params, dt)
        var = quant_cpp.ou.OUProcess.conditional_variance(params, dt)

        # Mean should be between x_t and theta
        assert mean < 0.5
        assert mean > 0.0

        # Variance should be positive
        assert var > 0

    def test_optimal_boundaries(self):
        """Test optimal trading boundaries."""
        params = quant_cpp.ou.OUParameters(0.0, 5.0, 0.1)

        lower, upper, exit_target = quant_cpp.ou.OUProcess.optimal_boundaries(
            params, 0.001, 0.05
        )

        # Entry boundaries should be symmetric around theta
        assert lower < params.theta
        assert upper > params.theta

        # Exit target should be near theta
        assert abs(exit_target - params.theta) < params.stationary_std()


class TestPythonWrappers:
    """Tests for Python wrapper classes."""

    def test_heston_wrapper(self):
        """Test Heston Python wrapper."""
        from quant_trading.models import HestonModel

        model = HestonModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)

        price = model.price_option(strike=100, maturity=1.0, spot=100,
                                   rate=0.05, dividend=0.02)

        assert price > 0
        assert isinstance(price, float)

    def test_heston_wrapper_vectorized(self):
        """Test Heston wrapper vectorized pricing."""
        from quant_trading.models import HestonModel

        model = HestonModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)

        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        prices = model.price_options(strikes, 1.0, 100.0, 0.05, 0.02)

        assert isinstance(prices, np.ndarray)
        assert len(prices) == len(strikes)
        assert np.all(prices > 0)

    def test_sabr_wrapper(self):
        """Test SABR Python wrapper."""
        from quant_trading.models import SABRModel

        model = SABRModel(beta=0.5)

        vol = model.implied_volatility(
            strike=105.0, forward=100.0, maturity=1.0,
            alpha=0.2, rho=-0.3, nu=0.4
        )

        assert vol > 0
        assert isinstance(vol, float)

    def test_ou_wrapper(self):
        """Test OU process Python wrapper."""
        from quant_trading.models import OUProcess, OUParameters

        params = OUParameters(theta=0.0, mu=5.0, sigma=0.1)

        path = OUProcess.simulate(params, x0=0.5, T=1.0, n_steps=252, seed=42)

        assert isinstance(path, np.ndarray)
        assert len(path) == 253

    def test_ou_fit_wrapper(self):
        """Test OU fitting wrapper."""
        from quant_trading.models import OUProcess, OUParameters

        params = OUParameters(theta=0.0, mu=5.0, sigma=0.1)
        path = OUProcess.simulate(params, x0=0.0, T=1.0, n_steps=252, seed=42)

        result = OUProcess.fit_mle(path, dt=1.0 / 252.0)

        assert result.converged
        assert isinstance(result.params, OUParameters)


class TestPDESolverBindings:
    """Tests for PDE solver C++ bindings."""

    def test_grid1d_basic(self):
        """Test Grid1D construction and basic operations."""
        grid = quant_cpp.solvers.Grid1D(0.0, 1.0, 101, False)
        assert grid.size() == 101
        assert abs(grid.min() - 0.0) < 1e-10
        assert abs(grid.max() - 1.0) < 1e-10
        assert abs(grid[0] - 0.0) < 1e-10
        assert abs(grid[100] - 1.0) < 1e-10

    def test_grid1d_log_space(self):
        """Test Grid1D in log-space."""
        grid = quant_cpp.solvers.Grid1D(1.0, 100.0, 101, True)
        assert grid.size() == 101
        assert abs(grid.min() - 1.0) < 1e-10
        assert abs(grid.max() - 100.0) < 1e-10
        # Log-space grid should be denser near x_min
        dx_start = grid[1] - grid[0]
        dx_end = grid[100] - grid[99]
        assert dx_start < dx_end

    def test_grid1d_find_index(self):
        """Test Grid1D index finding."""
        grid = quant_cpp.solvers.Grid1D(0.0, 100.0, 101, False)
        assert grid.find_index(50.0) == 50
        assert grid.find_index(0.0) == 0
        assert grid.find_index(100.0) == 100

    def test_grid1d_interpolate(self):
        """Test Grid1D interpolation."""
        grid = quant_cpp.solvers.Grid1D(0.0, 1.0, 11, False)
        # Create values = x^2
        values = np.array([grid[i]**2 for i in range(11)])
        # Interpolate at x=0.5 should give 0.25
        result = grid.interpolate(0.5, values)
        assert abs(result - 0.25) < 0.01

    def test_black_scholes_params_default(self):
        """Test BlackScholesPDEParams default construction."""
        params = quant_cpp.solvers.BlackScholesPDEParams()
        assert params.sigma == 0.2
        assert params.r == 0.05
        assert params.q == 0.0
        assert params.T == 1.0
        assert params.K == 100.0
        assert params.option_type == quant_cpp.solvers.OptionType.Call
        assert params.exercise == quant_cpp.solvers.ExerciseStyle.European

    def test_black_scholes_european_call(self):
        """Test Black-Scholes PDE for European call option."""
        params = quant_cpp.solvers.BlackScholesPDEParams()
        params.sigma = 0.2
        params.r = 0.05
        params.T = 1.0
        params.K = 100.0
        params.option_type = quant_cpp.solvers.OptionType.Call
        params.exercise = quant_cpp.solvers.ExerciseStyle.European
        params.n_space = 200
        params.n_time = 100

        solver = quant_cpp.solvers.BlackScholesPDESolver(params)
        result = solver.solve(100.0)

        # Compute analytical Black-Scholes price
        from scipy.stats import norm
        d1 = (np.log(100 / 100) + (0.05 + 0.5 * 0.2**2) * 1.0) / (0.2 * np.sqrt(1.0))
        d2 = d1 - 0.2 * np.sqrt(1.0)
        analytical = 100 * norm.cdf(d1) - 100 * np.exp(-0.05) * norm.cdf(d2)

        # Check price is close to analytical (within 1%)
        assert abs(result.price - analytical) / analytical < 0.01

        # Check delta is reasonable (ATM call delta ~0.5-0.7)
        assert 0.4 < result.delta < 0.8

        # Check gamma is positive
        assert result.gamma > 0

    def test_black_scholes_european_put(self):
        """Test Black-Scholes PDE for European put option."""
        params = quant_cpp.solvers.BlackScholesPDEParams()
        params.sigma = 0.2
        params.r = 0.05
        params.T = 1.0
        params.K = 100.0
        params.option_type = quant_cpp.solvers.OptionType.Put
        params.exercise = quant_cpp.solvers.ExerciseStyle.European
        params.n_space = 200
        params.n_time = 100

        solver = quant_cpp.solvers.BlackScholesPDESolver(params)
        result = solver.solve(100.0)

        # Compute analytical Black-Scholes price
        from scipy.stats import norm
        d1 = (np.log(100 / 100) + (0.05 + 0.5 * 0.2**2) * 1.0) / (0.2 * np.sqrt(1.0))
        d2 = d1 - 0.2 * np.sqrt(1.0)
        analytical = 100 * np.exp(-0.05) * norm.cdf(-d2) - 100 * norm.cdf(-d1)

        # Check price is close to analytical (within 1%)
        assert abs(result.price - analytical) / analytical < 0.01

    def test_black_scholes_put_call_parity(self):
        """Test put-call parity for Black-Scholes PDE."""
        params = quant_cpp.solvers.BlackScholesPDEParams()
        params.sigma = 0.2
        params.r = 0.05
        params.T = 1.0
        params.K = 100.0
        params.n_space = 200
        params.n_time = 100

        # Price call
        params.option_type = quant_cpp.solvers.OptionType.Call
        params.exercise = quant_cpp.solvers.ExerciseStyle.European
        call_solver = quant_cpp.solvers.BlackScholesPDESolver(params)
        call_result = call_solver.solve(100.0)

        # Price put
        params.option_type = quant_cpp.solvers.OptionType.Put
        put_solver = quant_cpp.solvers.BlackScholesPDESolver(params)
        put_result = put_solver.solve(100.0)

        # Put-Call Parity: C - P = S - K*exp(-rT)
        S, K, r, T = 100.0, 100.0, 0.05, 1.0
        expected_diff = S - K * np.exp(-r * T)
        actual_diff = call_result.price - put_result.price

        assert abs(actual_diff - expected_diff) < 0.5

    def test_black_scholes_american_put_premium(self):
        """Test American put has early exercise premium."""
        params = quant_cpp.solvers.BlackScholesPDEParams()
        params.sigma = 0.2
        params.r = 0.05
        params.T = 1.0
        params.K = 100.0
        params.option_type = quant_cpp.solvers.OptionType.Put
        params.n_space = 200
        params.n_time = 100

        # European put
        params.exercise = quant_cpp.solvers.ExerciseStyle.European
        eu_solver = quant_cpp.solvers.BlackScholesPDESolver(params)
        eu_result = eu_solver.solve(100.0)

        # American put
        params.exercise = quant_cpp.solvers.ExerciseStyle.American
        am_solver = quant_cpp.solvers.BlackScholesPDESolver(params)
        am_result = am_solver.solve(100.0)

        # American put should be worth at least as much as European
        assert am_result.price >= eu_result.price - 0.01

    def test_black_scholes_itm_otm(self):
        """Test ITM and OTM option pricing."""
        params = quant_cpp.solvers.BlackScholesPDEParams()
        params.sigma = 0.2
        params.r = 0.05
        params.T = 1.0
        params.K = 100.0
        params.option_type = quant_cpp.solvers.OptionType.Call
        params.exercise = quant_cpp.solvers.ExerciseStyle.European
        params.n_space = 200
        params.n_time = 100

        solver = quant_cpp.solvers.BlackScholesPDESolver(params)

        # ITM call (S > K)
        itm_result = solver.solve(120.0)
        # OTM call (S < K)
        otm_result = solver.solve(80.0)
        # ATM call (S = K)
        atm_result = solver.solve(100.0)

        # ITM > ATM > OTM for calls
        assert itm_result.price > atm_result.price > otm_result.price

    def test_hjb_params_default(self):
        """Test HJBParams default construction."""
        params = quant_cpp.solvers.HJBParams()
        assert params.theta == 0.0
        assert params.mu == 5.0
        assert params.sigma == 0.1
        assert params.r == 0.05

    def test_hjb_solver_basic(self):
        """Test basic HJB solver."""
        params = quant_cpp.solvers.HJBParams()
        params.theta = 0.0
        params.mu = 5.0
        params.sigma = 0.1
        params.r = 0.05
        params.c_entry = 0.001
        params.c_exit = 0.001
        params.T = 1.0
        params.problem = quant_cpp.solvers.StoppingProblem.EntryLong
        params.n_space = 200
        params.n_time = 200

        solver = quant_cpp.solvers.HJBSolver(params)
        result = solver.solve()

        # Value function should be computed
        assert len(result.value_function) == params.n_space
        assert len(result.x_grid) == params.n_space

        # For entry long, lower boundary should exist
        # (enter long when price is sufficiently below theta)
        assert result.lower_boundary is not None or result.upper_boundary is not None

    def test_hjb_value_function(self):
        """Test HJB value function properties."""
        params = quant_cpp.solvers.HJBParams()
        params.theta = 0.0
        params.mu = 5.0
        params.sigma = 0.1
        params.r = 0.05
        params.c_entry = 0.001
        params.c_exit = 0.001
        params.T = 1.0
        params.problem = quant_cpp.solvers.StoppingProblem.EntryLong
        params.n_space = 200
        params.n_time = 200

        solver = quant_cpp.solvers.HJBSolver(params)
        result = solver.solve()

        # Value at mean should be positive (expected profit from trading)
        assert result.value_at(0.0) >= 0

        # Value should be finite everywhere
        for x in [-0.3, -0.1, 0.0, 0.1, 0.3]:
            assert math.isfinite(result.value_at(x))

    def test_hjb_all_boundaries(self):
        """Test computing all optimal trading boundaries."""
        params = quant_cpp.solvers.HJBParams()
        params.theta = 0.0
        params.mu = 5.0
        params.sigma = 0.1
        params.r = 0.05
        params.c_entry = 0.001
        params.c_exit = 0.001
        params.T = 1.0
        params.n_space = 100
        params.n_time = 100

        solver = quant_cpp.solvers.HJBSolver(params)
        bounds = solver.solve_all_boundaries()

        # Entry long should be below theta
        assert bounds.entry_long < params.theta
        # Entry short should be above theta
        assert bounds.entry_short > params.theta
        # Exit long should be around theta
        assert abs(bounds.exit_long - params.theta) < 0.5
        # Exit short should be around theta
        assert abs(bounds.exit_short - params.theta) < 0.5

    def test_cfl_utility_functions(self):
        """Test CFL utility functions."""
        # Check CFL condition
        dt, dx, diffusion = 0.001, 0.01, 0.5
        is_stable = quant_cpp.solvers.check_cfl_condition(dt, dx, diffusion)
        assert isinstance(is_stable, bool)

        # Compute stable dt
        stable_dt = quant_cpp.solvers.compute_stable_dt(dx, diffusion, 0.9)
        assert stable_dt > 0
        assert quant_cpp.solvers.check_cfl_condition(stable_dt, dx, diffusion)
