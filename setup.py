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
        "Topic :: Scientific/Engineering",
    ],
    keywords="qaoa",
    packages=setuptools.find_packages(
        include=["qaoa_training_pipeline", "qaoa_training_pipeline.*"]
    ),
    install_requires=REQUIREMENTS,
    extras_require={
        # Torch-free AI-inference runtime (AIInference).
        # huggingface_hub lazily fetches the ONNX weights pinned in the manifest.
        "inference": ["onnxruntime", "huggingface_hub"],
    },
    include_package_data=True,
    # Ship the small per-bundle configs and the HF weight manifest. The large
    # model.onnx / model.onnx.data weights are lazily downloaded from the
    # HuggingFace Hub at first use (see inference/model_registry.py); the .onnx*
    # globs below keep them in the wheel only until that migration is cut over,
    # after which they should be removed.
    package_data={
        "qaoa_training_pipeline": [
            "inference/model_configs/hf_manifest.json",
            "inference/model_configs/*/*/model_config.json",
            "inference/model_configs/*/*/model.onnx",
            "inference/model_configs/*/*/model.onnx.data",
        ],
    },
    python_requires=">=3.10",
    zip_safe=False,
)
