"""
Setup script for Project EvoArchitect v3
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="project-evoarchitect-v3",
    version="3.0.0",
    author="EvoMaster AI",
    author_email="your.email@example.com",
    description="Autonomous AI-Evolutionary Agent for Neural Architecture Discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Project_EvoArchitect_v3",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "evoarchitect=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
