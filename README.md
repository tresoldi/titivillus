# Titivillus

<img src="https://raw.githubusercontent.com/tresoldi/titivillus/main/docs/titivillus.png" width="200" alt="Titivillus"/>

![CI](https://github.com/tresoldi/titivillus/workflows/CI/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/titivillus.svg)](https://pypi.org/project/titivillus)

`titivillus` is a Python library and related command-line tools for performing community detection on preprocessed orthographic tabular data. Using advanced clustering algorithms and dimensionality reduction techniques, `titivillus` helps uncover natural groupings within complex datasets.

The library is named after [Titivillus](https://en.wikipedia.org/wiki/Titivillus), a demon from medieval folklore who was said to introduce errors into the work of scribes. In contrast, `titivillus` seeks to bring order to data, revealing underlying structures that may seem as chaotic as a scribe's mistakes.

## How does `titivillus` work?

`titivillus` is structured around a simple workflow:

1. **Data Preparation**: Import data and preprocess it for analysis.
2. **Feature Scaling**: Standardize or normalize your data for optimal performance.
3. **Dimensionality Reduction**: Apply PCA to reduce the number of features, if necessary.
4. **Clustering**: Choose from various clustering algorithms to find patterns in data.
5. **Visualization**: Generate plots to visualize the clusters and data distribution.
6. **Output**: Export the clustered data with labels for further investigation.

## Installation

To install `titivillus`, run:

```bash
$ pip install titivillus
```

The installation will handle all required dependencies, including `pandas`, `numpy`, `scikit-learn`, and `matplotlib`. For best practices, create a virtual environment before installation.

## How to use

Here's a quick example to get you started with `titivillus`:

```python
from titivillus import cluster, visualize

# Load your dataset
data = ...

# Cluster your data
labels = cluster(data, method='kmeans', n_clusters=5)

# Visualize the results
visualize(data, labels, mode='2D')
```

For more detailed usage, please refer to the documentation and command-line help:

```bash
$ titivillus --help
```

## Changelog

- Version 0.1

  - Initial release with support for multiple clustering algorithms and visualization tools.

## Community Guidelines

Contributions are welcome! Please submit issues and pull requests on GitHub. Consult the `CONTRIBUTING.md` file for more information on how to contribute.

## Author and Citation

`titivillus` is developed by Tiago Tresoldi (tiago.tresoldi@lingfil.uu.se).

Citing `titivillus`:

> Tresoldi, Tiago (2023). Titivillus: A Python library for detecting and visualizing communities in steammatological data. Version 0.1. Uppsala University.

BibTeX entry:

```bibtex
@misc{Tresoldi2023titivillus,
  author = {Tresoldi, Tiago},
  title = {Titivillus: A Python library for detecting and visualizing communities in stemmatological data},
  howpublished = {\url{https://github.com/tresoldi/titivillus}},
  address = {Uppsala University},
  year = {2023},
}
```