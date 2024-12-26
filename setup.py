from setuptools import setup, find_packages

setup(
    name="kkexpr",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "datafeed.rust_expr": ["*.dll", "*.so", "*.dylib"],
    },
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
    ],
    python_requires=">=3.8",
) 