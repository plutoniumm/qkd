from setuptools import find_packages, setup

# No pyproject.toml: setup_requires (below) tells the build frontend to install
# setuptools-rust into the build env, then re-run this file with it importable.
# The first pass runs before that happens, so guard the import.
try:
    from setuptools_rust import Binding, RustExtension

    rust_extensions = [
        RustExtension(
            "qkd._core",
            path="Cargo.toml",
            binding=Binding.PyO3,
            py_limited_api=True,
        )
    ]
except ImportError:
    rust_extensions = []

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qkd",
    version="0.2.0",
    description=(
        "Continuous-variable quantum optics and CV-QKD simulator: Gaussian "
        "symplectic core, pilot-assisted DSP, and hardware-parameterised key rates."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="plutoniumm",
    author_email="haskell-game@manav.ch",
    url="https://github.com/plutoniumm/qkd",
    license="MIT",
    keywords=[
        "quantum",
        "continuous-variable",
        "quantum-optics",
        "cvqkd",
        "qkd",
        "gaussian",
        "simulator",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.11",
    packages=find_packages(include=["qkd", "qkd.*"]),
    install_requires=["numpy"],
    extras_require={
        "dev": ["setuptools-rust", "twine", "black", "wheel"],
    },
    setup_requires=["setuptools-rust"],
    rust_extensions=rust_extensions,
    # Tag wheels cp311-abi3 (PEP 384 stable ABI) regardless of the building
    # Python, so one wheel per platform serves every Python >= 3.11. Matches the
    # abi3-py311 pyo3 feature; without this bdist_wheel would tag version-specific
    # wheels (e.g. cp314-cp314 when built under 3.14).
    options={"bdist_wheel": {"py_limited_api": "cp311"}},
    zip_safe=False,
)
