from setuptools import find_packages, setup

install_requires = [
    "numpy",
    "tensorflow<",
    "matplotlib",
    "pandas",
    "pytest",
]
setup(
    name="veos",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    author="Binjian Xin",
    author_email="binjian.xin@newrizon.com",
    entrypoints={
        "console_scripts": [
            "voes = veos.__main__:main",
        ],
    },
    description="A python package for vehicle energy optimization system",
)
