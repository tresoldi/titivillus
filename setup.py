"""
Setup file for titivillus.
"""

# Import Python standard libraries
import pathlib

from setuptools import setup

# The directory containing this file
LOCAL_PATH = pathlib.Path(__file__).parent

# The text of the README file
README_FILE = (LOCAL_PATH / "README.md").read_text()

# Load requirements, so they are listed in a single place
with open("requirements.txt", encoding="utf-8") as fp:
    install_requires = [dep.strip() for dep in fp.readlines()]

# This call to setup() does all the work
setup(
    author="Tiago Tresoldi",
    author_email="tiago.tresoldi@lingfil.uu.se",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
    ],
    description="A Python library for detecting and visualizing communities in steammatological data.",
    entry_points={
        "console_scripts": [
            "titivillus=titivillus.__main__:main",
            "x2titivillus=titivillus.__x2titivillus__:main",
        ]
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords=["community detection", "clustering", "networks", "steammatology"],
    license="GPLv3",
    long_description=README_FILE,
    long_description_content_type="text/markdown",
    name="titivillus",
    packages=["titivillus"],
    python_requires=">=3.8",
    test_suite="tests",
    tests_require=[],
    url="https://github.com/tresoldi/titivillus",
    version="0.1",
    zip_safe=False,
)
