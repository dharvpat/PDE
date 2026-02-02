/**
 * @file pde_bindings.cpp
 * @brief Python bindings for PDE solvers using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include "solvers/pde_core.hpp"
#include "solvers/black_scholes_pde.hpp"
#include "solvers/heston_pde.hpp"
#include "solvers/hjb_solver.hpp"

namespace py = pybind11;

void init_pde_bindings(py::module_& m) {
    using namespace quant::solvers;

    // Create solvers submodule
    auto solvers = m.def_submodule("solvers", "PDE solvers for option pricing");

    // Enums
    py::enum_<OptionType>(solvers, "OptionType")
        .value("Call", OptionType::Call)
        .value("Put", OptionType::Put);

    py::enum_<ExerciseStyle>(solvers, "ExerciseStyle")
        .value("European", ExerciseStyle::European)
        .value("American", ExerciseStyle::American);

    py::enum_<TimeScheme>(solvers, "TimeScheme")
        .value("Explicit", TimeScheme::Explicit)
        .value("Implicit", TimeScheme::Implicit)
        .value("CrankNicolson", TimeScheme::CrankNicolson)
        .value("ADI", TimeScheme::ADI);

    py::enum_<StoppingProblem>(solvers, "StoppingProblem")
        .value("EntryLong", StoppingProblem::EntryLong)
        .value("EntryShort", StoppingProblem::EntryShort)
        .value("ExitLong", StoppingProblem::ExitLong)
        .value("ExitShort", StoppingProblem::ExitShort);

    // Grid1D
    py::class_<Grid1D>(solvers, "Grid1D",
        "1D spatial grid for PDE discretization")
        .def(py::init<double, double, size_t, bool>(),
             py::arg("x_min"), py::arg("x_max"),
             py::arg("n_points"), py::arg("use_log_space") = false)
        .def("size", &Grid1D::size)
        .def("__getitem__", &Grid1D::operator[])
        .def("dx", &Grid1D::dx)
        .def("points", &Grid1D::points)
        .def("min", &Grid1D::min)
        .def("max", &Grid1D::max)
        .def("find_index", &Grid1D::find_index)
        .def("interpolate", &Grid1D::interpolate);

    // Grid2D
    py::class_<Grid2D>(solvers, "Grid2D",
        "2D spatial grid for 2D PDEs")
        .def(py::init<const Grid1D&, const Grid1D&>())
        .def("nx", &Grid2D::nx)
        .def("ny", &Grid2D::ny)
        .def("size", &Grid2D::size)
        .def("to_linear", &Grid2D::to_linear)
        .def("to_2d", &Grid2D::to_2d)
        .def("x", &Grid2D::x)
        .def("y", &Grid2D::y);

    // Black-Scholes PDE params
    py::class_<BlackScholesPDEParams>(solvers, "BlackScholesPDEParams",
        "Parameters for Black-Scholes PDE solver")
        .def(py::init<>())
        .def_readwrite("sigma", &BlackScholesPDEParams::sigma)
        .def_readwrite("r", &BlackScholesPDEParams::r)
        .def_readwrite("q", &BlackScholesPDEParams::q)
        .def_readwrite("T", &BlackScholesPDEParams::T)
        .def_readwrite("K", &BlackScholesPDEParams::K)
        .def_readwrite("option_type", &BlackScholesPDEParams::option_type)
        .def_readwrite("exercise", &BlackScholesPDEParams::exercise)
        .def_readwrite("n_space", &BlackScholesPDEParams::n_space)
        .def_readwrite("n_time", &BlackScholesPDEParams::n_time)
        .def_readwrite("s_min_mult", &BlackScholesPDEParams::s_min_mult)
        .def_readwrite("s_max_mult", &BlackScholesPDEParams::s_max_mult)
        .def_readwrite("scheme", &BlackScholesPDEParams::scheme);

    // Black-Scholes PDE result
    py::class_<BlackScholesPDEResult>(solvers, "BlackScholesPDEResult",
        "Result of Black-Scholes PDE solution")
        .def_readonly("price", &BlackScholesPDEResult::price)
        .def_readonly("delta", &BlackScholesPDEResult::delta)
        .def_readonly("gamma", &BlackScholesPDEResult::gamma)
        .def_readonly("theta", &BlackScholesPDEResult::theta)
        .def_readonly("prices", &BlackScholesPDEResult::prices)
        .def_readonly("spot_grid", &BlackScholesPDEResult::spot_grid)
        .def_readonly("early_exercise_optimal", &BlackScholesPDEResult::early_exercise_optimal);

    // Black-Scholes PDE solver
    py::class_<BlackScholesPDESolver>(solvers, "BlackScholesPDESolver",
        "Black-Scholes PDE solver using finite differences")
        .def(py::init<const BlackScholesPDEParams&>())
        .def("solve", &BlackScholesPDESolver::solve, py::arg("S0"),
             "Solve the Black-Scholes PDE and return price with Greeks");

    // Heston PDE params
    py::class_<HestonPDEParams>(solvers, "HestonPDEParams",
        "Parameters for Heston PDE solver")
        .def(py::init<>())
        .def_readwrite("kappa", &HestonPDEParams::kappa)
        .def_readwrite("theta", &HestonPDEParams::theta)
        .def_readwrite("sigma", &HestonPDEParams::sigma)
        .def_readwrite("rho", &HestonPDEParams::rho)
        .def_readwrite("v0", &HestonPDEParams::v0)
        .def_readwrite("r", &HestonPDEParams::r)
        .def_readwrite("q", &HestonPDEParams::q)
        .def_readwrite("T", &HestonPDEParams::T)
        .def_readwrite("K", &HestonPDEParams::K)
        .def_readwrite("option_type", &HestonPDEParams::option_type)
        .def_readwrite("exercise", &HestonPDEParams::exercise)
        .def_readwrite("n_spot", &HestonPDEParams::n_spot)
        .def_readwrite("n_vol", &HestonPDEParams::n_vol)
        .def_readwrite("n_time", &HestonPDEParams::n_time)
        .def_readwrite("s_min_mult", &HestonPDEParams::s_min_mult)
        .def_readwrite("s_max_mult", &HestonPDEParams::s_max_mult)
        .def_readwrite("v_max", &HestonPDEParams::v_max);

    // Heston PDE result
    py::class_<HestonPDEResult>(solvers, "HestonPDEResult",
        "Result of Heston PDE solution")
        .def_readonly("price", &HestonPDEResult::price)
        .def_readonly("delta", &HestonPDEResult::delta)
        .def_readonly("gamma", &HestonPDEResult::gamma)
        .def_readonly("vega", &HestonPDEResult::vega)
        .def_readonly("theta", &HestonPDEResult::theta)
        .def_readonly("prices", &HestonPDEResult::prices)
        .def_readonly("spot_grid", &HestonPDEResult::spot_grid)
        .def_readonly("vol_grid", &HestonPDEResult::vol_grid);

    // Heston PDE solver
    py::class_<HestonPDESolver>(solvers, "HestonPDESolver",
        "Heston stochastic volatility PDE solver using ADI method")
        .def(py::init<const HestonPDEParams&>())
        .def("solve", &HestonPDESolver::solve, py::arg("S0"),
             "Solve the Heston PDE and return price with Greeks");

    // HJB params
    py::class_<HJBParams>(solvers, "HJBParams",
        "Parameters for HJB optimal stopping solver")
        .def(py::init<>())
        .def_readwrite("theta", &HJBParams::theta)
        .def_readwrite("mu", &HJBParams::mu)
        .def_readwrite("sigma", &HJBParams::sigma)
        .def_readwrite("r", &HJBParams::r)
        .def_readwrite("c_entry", &HJBParams::c_entry)
        .def_readwrite("c_exit", &HJBParams::c_exit)
        .def_readwrite("T", &HJBParams::T)
        .def_readwrite("problem", &HJBParams::problem)
        .def_readwrite("n_space", &HJBParams::n_space)
        .def_readwrite("n_time", &HJBParams::n_time)
        .def_readwrite("x_min", &HJBParams::x_min)
        .def_readwrite("x_max", &HJBParams::x_max);

    // HJB result
    py::class_<HJBResult>(solvers, "HJBResult",
        "Result of HJB optimal stopping solution")
        .def_readonly("value_function", &HJBResult::value_function)
        .def_readonly("x_grid", &HJBResult::x_grid)
        .def_property_readonly("lower_boundary",
            [](const HJBResult& r) -> py::object {
                if (r.lower_boundary) return py::cast(*r.lower_boundary);
                return py::none();
            })
        .def_property_readonly("upper_boundary",
            [](const HJBResult& r) -> py::object {
                if (r.upper_boundary) return py::cast(*r.upper_boundary);
                return py::none();
            })
        .def("value_at", &HJBResult::value_at)
        .def("should_stop", &HJBResult::should_stop);

    // Optimal trading boundaries
    py::class_<OptimalTradingBoundaries>(solvers, "OptimalTradingBoundaries",
        "Optimal boundaries for mean-reversion trading")
        .def_readonly("entry_long", &OptimalTradingBoundaries::entry_long)
        .def_readonly("entry_short", &OptimalTradingBoundaries::entry_short)
        .def_readonly("exit_long", &OptimalTradingBoundaries::exit_long)
        .def_readonly("exit_short", &OptimalTradingBoundaries::exit_short)
        .def_readonly("stop_loss_long", &OptimalTradingBoundaries::stop_loss_long)
        .def_readonly("stop_loss_short", &OptimalTradingBoundaries::stop_loss_short);

    // HJB solver
    py::class_<HJBSolver>(solvers, "HJBSolver",
        "Hamilton-Jacobi-Bellman solver for optimal stopping")
        .def(py::init<const HJBParams&>())
        .def("solve", &HJBSolver::solve,
             "Solve the HJB equation for optimal stopping")
        .def("solve_all_boundaries", &HJBSolver::solve_all_boundaries,
             "Compute all optimal trading boundaries");

    // Utility functions
    solvers.def("check_cfl_condition", &check_cfl_condition,
        py::arg("dt"), py::arg("dx"), py::arg("diffusion"), py::arg("advection") = 0.0,
        "Check CFL stability condition for explicit schemes");

    solvers.def("compute_stable_dt", &compute_stable_dt,
        py::arg("dx"), py::arg("diffusion"), py::arg("cfl_factor") = 0.9,
        "Compute suggested stable time step based on CFL condition");
}
