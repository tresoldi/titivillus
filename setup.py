#!/usr/bin/env python3
"""
Setup configuration for Titivillus synthetic phylogenetic data generator.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read version from __init__.py
def get_version():
    version_file = os.path.join(this_directory, 'titivillus', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '0.1.0'

setup(
    name='titivillus',
    version=get_version(),
    author='Titivillus Development Team',
    author_email='contact@titivillus.org',
    description='Synthetic phylogenetic data generator for algorithm testing and validation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/titivillus/titivillus',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    keywords='phylogenetics synthetic-data bioinformatics linguistics cultural-evolution',
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'PyYAML>=6.0',
        'click>=8.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=1.0.0',
        ],
        'plotting': [
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
        ],
        'fast': [
            'numba>=0.56.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'titivillus=titivillus.cli:main',
        ],
    },
    package_data={
        'titivillus': [
            'config/*.yaml',
            'examples/*.yaml',
            'templates/*.yaml',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)