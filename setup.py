"""Make src/ importable as zsslr_azsl after `pip install -e .`"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="zsslr-azsl",
    version="1.0.0",
    description="Multimodal Zero-Shot Sign Language Recognition for AzSL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lala2398/ZSSLR-AzSL",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)
