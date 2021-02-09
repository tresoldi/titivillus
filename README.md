# Titivillus

![Titivillus](https://raw.githubusercontent.com/tresoldi/titivillus/main/docs/titivillus.png)

![CI](https://github.com/tresoldi/titivillus/workflows/CI/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/titivillus.svg)](https://pypi.org/project/titivillus)

`titivillus` is a Python library and related command-line tools for simulating 
stemmatological networks and related data (characters, states, edge length etc.). It 
is intended for benchmarking quantitative methods of textual evolution, also providing 
dummy tree and networks for their development and debugging.

The library is named after [Titivillus](https://en.wikipedia.org/wiki/Titivillus), a 
demon commonly referenced to in Medieval times and said to work on behalf of Belphegor,
Lucifer, or Satan to introduce errors 
into the work of scribes. It can be compared to the twentieth-century folkloric 
mischievous creature that causes mechanical failures, the gremlin.

## How does `titivillus` work?

The library offers a number of abstractions suited for the study of textual evolution, 
in particular without forcing a purely arboreal evolution. Each *codex* carries a 
number of independent *characters*t, each with its own history.

Where applicable, the random generation follows that of another package released by 
the author for the simulation of phylogenetic data, [ngesh](https://pypi.org/project/ngesh/).

## Installation

In any standard Python environment, `titivillus` can be installed with

```bash
$ pip install titivillus
```

The `pip` installation will automatically fetch dependencies such as `numpy` and 
`networkx`, if necessary. It is highly recommendend that the library is installed in 
its own virtual environment.

## How to use

For most usages, the creation of a random stemma can be easily performed from Python with:

```python
import titivillus
stemma = titivillus.random_stemma()
```

Among the various parameters, it is possible to pass a pseudo-random number generator 
seed that guarantees reproducibility across different calls. 

```python
stemma2 = titivillus.random_stemma(seed="uppsala")
```

The contents of the stemma can be inspected following the available tests. A graphical 
version, using `networkx`, can be obtained with:

```python
import matplotlib.pyplot as plt
import networkx as nx

graph = stemma2.as_graph()
nx.draw(graph)
plt.show()
```


![random stemma](https://raw.githubusercontent.com/tresoldi/titivillus/main/docs/graph1.png)

No stand-alone command-line tool has been released yet.

## Changelog

Version 0.0.1:

  - First public release, aligned with experiments for the Apophthegmata Patrum

## Community guidelines

While the author can be contacted directly for support, it is recommended that third 
parties use GitHub standard features, such as issues and pull requests, to contribute, 
report problems, or seek support.

Contributing guidelines, including a code of conduct, can be found in the
`CONTRIBUTING.md` file.

## Author and citation

The library is developed by Tiago Tresoldi (tiago.tresoldi@lingfil.uu.se).

If you use `titivillus`, please cite it as:

> Tresoldi, Tiago (2021). Titivillus, a tool for simulating random stemmatological 
> networks. Version 0.0.1. Uppsala. Available at: https://github.com/tresoldi/titivillus

In BibTeX:

```
@misc{Tresoldi2021titivillus,
  author = {Tresoldi, Tiago},
  title = {Titivillus, a tool for simulating random stemmatological networks},
  howpublished = {\url{https://github.com/tresoldi/titivillus}},
  address = {Uppsala},
  year = {2021},
}