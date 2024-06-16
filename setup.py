from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sudoku",
    version="0.2.3",
    description="Sudoku",
    long_description=long_description,
    author="markduan",
    packages=find_packages(exclude=["tests"]),
    license="GPL",
    python_requires=">=3.8",
    install_requires=["opencv-python>=4.9.0.80", "tensorflow>=2.13.1"],
    entry_points={"console_scripts": ["sudoku = sudoku.__main__:main"]},
)
