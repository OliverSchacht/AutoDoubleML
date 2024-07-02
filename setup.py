from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='AutoDoubleML',
    version='0.0.9000',
    description='Additional DoubleML classes to integrate AutoML frameworks in DoubleML (work in progress).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Oliver Schacht, Philipp Bach',
    author_email='oliver.schacht@uni-hamburg.de',
    packages=find_packages(),
    install_requires=[
        'DoubleML',
        'numpy',
        'pandas',
    ],
    include_package_data=True,
    python_requires='>=3.9',
)