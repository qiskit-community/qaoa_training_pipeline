import os
import setuptools

long_description = """A pipeline to generate good parameters for QAOA Ansatz circuits."""

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

VERSION_PATH = os.path.join(os.path.dirname(__file__), "qaoa_training_pipeline", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

setuptools.setup(
    name="qaoa_training_pipeline",
    version=VERSION,
    description="QAOA training pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qiskit-community/qaoa_training_pipeline",
    author="Daniel Egger, Elena Pena Tapia, Alberto Baiardi",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qaoa",
    packages=setuptools.find_packages(
        include=["qaoa_training_pipeline", "qaoa_training_pipeline.*"]
    ),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.8",
    zip_safe=False,
)
