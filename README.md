# Titivillus

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/titivillus.svg)](https://badge.fury.io/py/titivillus)

**Titivillus** is a comprehensive synthetic phylogenetic data generator designed for rigorous testing and validation of phylogenetic inference algorithms. Named after the demon of scribal errors, it generates realistic datasets with known ground truth across multiple evolutionary domains.

## 🌟 Features

### Multi-Domain Support
- **Linguistics**: Phonological, morphological, and lexical character evolution
- **Stemmatology**: Manuscript tradition analysis with scribal errors
- **Cultural Evolution**: Innovation, diffusion, and cultural trait transmission

### Realistic Evolutionary Modeling
- **Asymmetric Evolution**: Directional biases and unequal transition rates
- **Network Effects**: Reticulation, borrowing, and contact zones
- **Domain-Specific Processes**: Tailored models for each evolutionary domain

### Comprehensive Error Injection
- **Missing Data Patterns**: Random, systematic, clustered, and taxon-specific
- **Measurement Errors**: Realistic observational noise and uncertainty
- **Polymorphisms**: Multiple states and ambiguous observations

### Ground Truth Tracking
- **Complete Provenance**: Full evolutionary histories for every character
- **Validation Metrics**: Comprehensive data for algorithm performance assessment
- **Multiple Validation Levels**: From basic to exhaustive ground truth

### Multi-Format Output
- **NEXUS**: Compatible with PAUP*, MrBayes, and other phylogenetic software
- **CSV**: Data matrices for statistical analysis in R/Python
- **JSON**: Complete metadata and machine-readable provenance
- **Newick**: Standard tree format with branch lengths

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install titivillus

# Install with optional dependencies
pip install titivillus[plotting,fast]

# Development installation
git clone https://github.com/titivillus/titivillus.git
cd titivillus
pip install -e .[dev]
```

### Command Line Usage

```bash
# Generate linguistic dataset
titivillus generate --template linguistic --taxa 20 --characters 50 --output ./data

# Generate from configuration file
titivillus generate --config my_config.yaml --output ./results

# Create example configuration
titivillus init --output config.yaml --template multidomain

# Validate configuration
titivillus validate --config config.yaml

# List available templates
titivillus templates --list
```

### Python API

```python
from titivillus import Generator, Config

# Quick linguistic dataset
generator = Generator.quick_linguistic(taxa_count=25, character_count=60)
dataset = generator.generate()
generator.export(dataset, "./output")

# From YAML configuration
generator = Generator.from_yaml("config.yaml")
dataset, output_paths = generator.generate_and_export("./results")

# Multi-domain dataset
generator = Generator.quick_multidomain(taxa_count=30)
dataset = generator.generate()
print(f"Generated {len(dataset.taxa)} taxa with {len(dataset.characters)} characters")
```

## 📖 Documentation

### Configuration

Titivillus uses YAML configuration files for flexible dataset specification:

```yaml
name: "linguistic_test_dataset"
description: "Synthetic linguistic data for phylogenetic testing"
seed: 42

tree:
  taxa_count: 25
  tree_height: 1.0
  tree_shape: "balanced"

active_domains:
  - "linguistics"

domains:
  linguistics:
    character_count: 80
    substitution_rate: 1.2
    asymmetric_bias: 0.7
    borrowability_range: [0.1, 0.8]
    phonological_constraints: true

networks:
  enable_reticulation: true
  reticulation_rate: 0.15
  contact_zones: 3
  borrowing_rate: 0.25

errors:
  missing_data_rate: 0.18
  measurement_error_rate: 0.06
  polymorphism_rate: 0.10

output:
  formats: ["nexus", "csv", "json"]
  base_name: "synthetic_linguistic"
  include_timestamp: true
```

### Domain-Specific Features

#### Linguistics
- **Sound Change Modeling**: Realistic phonological evolution patterns
- **Borrowing Effects**: Contact-induced character transmission
- **Areal Features**: Geographic influence on linguistic traits
- **Morphological Complexity**: Paradigmatic and syntagmatic relationships

#### Stemmatology  
- **Scribal Error Patterns**: Realistic copying mistakes and corrections
- **Variant Complexity**: Multiple textual readings and corrections
- **Contamination Events**: Cross-influence between manuscript traditions
- **Paleographic Confusion**: Letter similarity-based errors

#### Cultural Evolution
- **Innovation vs. Inheritance**: Balance between novel traits and transmission
- **Prestige Effects**: High-status variant preferences
- **Trade Route Transmission**: Cultural exchange via economic networks
- **Horizontal Transfer**: Cross-cultural trait adoption

### Network Effects

Titivillus supports sophisticated phylogenetic network modeling:

- **Reticulation Events**: Hybridization and convergent evolution
- **Contact Zones**: Geographic regions of intensive interaction
- **Borrowing Networks**: Directed transmission between lineages
- **Migration Patterns**: Population movement and gene flow

## 🔬 Use Cases

### Algorithm Development
- **Method Validation**: Test new phylogenetic inference algorithms
- **Performance Benchmarking**: Compare methods with known ground truth
- **Edge Case Testing**: Evaluate robustness with challenging datasets
- **Parameter Sensitivity**: Assess method behavior across parameter spaces

### Research Applications
- **Simulation Studies**: Generate data for comparative phylogenetic analysis
- **Method Evaluation**: Assess existing software performance
- **Teaching**: Demonstrate phylogenetic concepts with controlled examples
- **Software Development**: Test phylogenetic analysis pipelines

### Specialized Domains
- **Historical Linguistics**: Test linguistic phylogeny reconstruction
- **Textual Criticism**: Validate manuscript tradition analysis
- **Cultural Phylogenetics**: Evaluate cultural evolution methods
- **Comparative Biology**: Generate test data for biological applications

## 🎯 Design Philosophy

Titivillus embraces the **messy reality** of phylogenetic data while providing **complete ground truth** for rigorous validation. Key principles:

### Realistic Complexity
Real phylogenetic data is messy, incomplete, and shaped by complex evolutionary processes. Titivillus generates data that reflects this complexity rather than oversimplified models.

### Comprehensive Ground Truth
Every aspect of data generation is tracked and recorded, enabling thorough validation of phylogenetic methods and deep understanding of their behavior.

### Domain Expertise
Each supported domain (linguistics, stemmatology, cultural evolution) incorporates domain-specific knowledge and realistic evolutionary processes.

### Software Compatibility
Generated datasets work seamlessly with existing phylogenetic software (PAUP*, MrBayes, BEAST, etc.) and analysis pipelines.

## 📊 Examples

### Linguistic Phylogeny
```bash
# Generate Indo-European-like dataset
titivillus generate --template linguistic \
  --taxa 30 --characters 100 \
  --output ./indo_european_test

# With network effects
titivillus init --output ie_config.yaml --template linguistic
# Edit config to add contact zones and borrowing
titivillus generate --config ie_config.yaml --output ./ie_network_test
```

### Manuscript Tradition
```bash
# Generate manuscript stemma
titivillus generate --template stemmatology \
  --taxa 15 --characters 200 \
  --output ./manuscript_tradition

# High error rate scenario
titivillus generate --template stemmatology \
  --taxa 20 --characters 150 \
  --output ./noisy_manuscripts \
  --config high_error_config.yaml
```

### Multi-Domain Analysis
```bash
# Combined linguistic and cultural data
titivillus generate --template multidomain \
  --taxa 40 --output ./combined_analysis

# Custom multi-domain configuration
titivillus init --output multi_config.yaml --template multidomain
titivillus generate --config multi_config.yaml --output ./custom_multidomain
```

## 🛠️ Development

### Contributing
We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Testing
```bash
# Run test suite
pytest tests/

# With coverage
pytest --cov=titivillus tests/

# Fast tests only
pytest -m "not slow" tests/
```

### Code Quality
```bash
# Format code
black titivillus/ tests/

# Type checking
mypy titivillus/

# Linting
flake8 titivillus/
```

## 📚 Citation

If you use Titivillus in your research, please cite:

```bibtex
@software{titivillus,
  title = {Titivillus: Synthetic Phylogenetic Data Generator},
  author = {Titivillus Development Team},
  url = {https://github.com/titivillus/titivillus},
  version = {0.1.0},
  year = {2024}
}
```

## 🤝 Acknowledgments

Named after **Titivillus**, the medieval demon said to collect errors made by scribes and cause spelling mistakes in religious texts. The name reflects our philosophy that realistic synthetic data must include the imperfections that make phylogenetic inference challenging.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [https://github.com/titivillus/titivillus/wiki](https://github.com/titivillus/titivillus/wiki)
- **Issue Tracker**: [https://github.com/titivillus/titivillus/issues](https://github.com/titivillus/titivillus/issues)
- **PyPI Package**: [https://pypi.org/project/titivillus/](https://pypi.org/project/titivillus/)
- **Discussions**: [https://github.com/titivillus/titivillus/discussions](https://github.com/titivillus/titivillus/discussions)