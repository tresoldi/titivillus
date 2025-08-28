# Titivillus Examples

This directory contains example configurations demonstrating various features of the Titivillus synthetic phylogenetic data generator.

## Available Examples

### `basic_linguistic.yaml`
A simple linguistic phylogenetic dataset suitable for testing:
- 15 taxa with 50 linguistic characters
- Balanced tree topology
- Moderate asymmetric evolution (bias = 0.6)
- Basic network effects (reticulation rate = 0.1)
- Low-moderate error rates

**Usage:**
```bash
titivillus generate --config examples/basic_linguistic.yaml --output ./linguistic_test
```

### `multidomain_analysis.yaml`
Comprehensive multi-domain dataset combining linguistic, stemmatological, and cultural data:
- 25 taxa across all three domains
- 140 total characters (60 linguistic + 45 stemmatology + 35 cultural)
- Enhanced network effects with multiple contact zones
- Realistic error patterns including domain-specific effects
- Comprehensive ground truth tracking

**Usage:**
```bash
titivillus generate --config examples/multidomain_analysis.yaml --output ./multidomain_test
```

## Quick Commands

Generate datasets using these examples:

```bash
# Basic linguistic test
titivillus generate --config examples/basic_linguistic.yaml --output ./data/linguistic

# Multi-domain analysis
titivillus generate --config examples/multidomain_analysis.yaml --output ./data/multidomain

# Validate configurations
titivillus validate --config examples/basic_linguistic.yaml
titivillus validate --config examples/multidomain_analysis.yaml
```

## Customization

These examples serve as starting points. Key parameters to adjust:

### Tree Structure
- `taxa_count`: Number of taxa (species/languages/manuscripts)
- `tree_height`: Maximum evolutionary distance
- `tree_shape`: balanced, pectinate, or random topology

### Characters
- `character_count`: Number of characters per domain
- `asymmetric_bias`: Degree of directional evolution (0.0 = symmetric, 1.0 = fully directional)
- `substitution_rate`: Rate of character evolution

### Network Effects
- `reticulation_rate`: Frequency of hybridization/borrowing events
- `contact_zones`: Number of geographic contact regions
- `borrowing_rate`: Rate of horizontal character transfer

### Data Quality
- `missing_data_rate`: Proportion of missing observations
- `measurement_error_rate`: Frequency of observational errors
- `polymorphism_rate`: Rate of ambiguous character states

## Creating Custom Configurations

Use the init command to create customizable templates:

```bash
# Create basic template
titivillus init --output my_config.yaml --template basic

# Create domain-specific templates  
titivillus init --output linguistic_custom.yaml --template linguistic
titivillus init --output stemmatology_custom.yaml --template stemmatology
titivillus init --output multidomain_custom.yaml --template multidomain
```

Then edit the YAML files to customize parameters for your specific research needs.

## Output Files

Each configuration generates multiple output formats:

- **NEXUS (.nex)**: Compatible with PAUP*, MrBayes, and other phylogenetic software
- **CSV (.csv)**: Character data matrices for statistical analysis
- **JSON (.json)**: Complete metadata and ground truth information
- **Newick (.tre)**: True phylogenetic tree with branch lengths

The ground truth information enables comprehensive validation of phylogenetic inference results.