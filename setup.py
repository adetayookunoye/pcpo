
from setuptools import setup, find_packages

setup(
    name="provable_prob_operator",
    version="0.1.0",
    description="Provably Constrained Probabilistic Operator Learning (JAX)",
    packages=find_packages(where="."),
    package_dir={"": "."},
    include_package_data=True,
    install_requires=[
        "jax[cpu]>=0.4.26",
        "optax>=0.2.1",
        "h5py>=3.10.0",
        "equinox>=0.11.5",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "matplotlib>=3.8.0",
        "tqdm>=4.66.0",
        "einops>=0.7.0",
        "pyyaml>=6.0.1",
        "chex>=0.1.86",
        "pytest>=8.0.0",
        "coverage>=7.4.0",
        "pylint>=3.2.0"
    ],
    entry_points={
        "console_scripts": [
            "ppo-train=src.train:main",
            "ppo-eval=src.eval:main",
            "ppo-download=src.data.fetch_preprocess:main"
        ]
    },
    python_requires=">=3.10",
)
