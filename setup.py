from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cse6242_team33_food_processing",
    version="0.1.0",
    author="Team 33",
    description="Food Processing Analysis for CSE 6242",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ViviQ600/CSE6242_team33_food_processing",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
