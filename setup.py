"""
Setup script for quant_trading package.

Builds C++ extensions via CMake and pybind11.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake extension for building C++ code."""

    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    """Custom build_ext command that runs CMake."""

    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-resolve to source directory
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DBUILD_TESTS=OFF",  # Don't build tests during pip install
        ]
        build_args = []

        # Compiler specification
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # Platform-specific settings
        if platform.system() == "Darwin":
            # Cross-compilation for macOS
            archs = os.environ.get("CMAKE_OSX_ARCHITECTURES", "")
            if archs:
                cmake_args += [f"-DCMAKE_OSX_ARCHITECTURES={archs}"]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control number of build jobs
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # Self-limit parallel builds to avoid memory issues
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
            check=True,
        )

    def run(self) -> None:
        """Run the build."""
        try:
            subprocess.run(["cmake", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError("CMake must be installed to build this package") from e

        super().run()


# Check if C++ extensions should be built
# Skip if SKIP_CPP_BUILD env var is set or if bindings don't exist yet
BUILD_CPP = (
    os.environ.get("SKIP_CPP_BUILD", "0") != "1"
    and Path("src/cpp/bindings/python_bindings.cpp").exists()
)

if BUILD_CPP:
    ext_modules = [CMakeExtension("quant_trading._cpp")]
    cmdclass = {"build_ext": CMakeBuild}
else:
    # Pure Python install - C++ extensions will be added later
    ext_modules = []
    cmdclass = {}

# Main setup
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
