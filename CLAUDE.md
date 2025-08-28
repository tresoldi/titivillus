# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the Titivillus synthetic phylogenetic data generator.

## Project Overview

Titivillus (named after the demon of scribal errors) is an independent library for generating synthetic phylogenetic datasets with realistic evolutionary processes, network effects, and data quality issues. It is designed specifically for testing and validating phylogenetic inference algorithms across multiple domains.

### Core Philosophy

Titivillus embraces the inherent messiness of real phylogenetic data while providing complete ground truth for rigorous algorithm testing. The name acknowledges that realistic synthetic data must include the imperfections that make phylogenetic inference challenging - just as the medieval demon Titivillus was said to introduce errors into manuscripts during copying.

When given a commit message, use ONLY AND EXACTLY that commit message.

## Development Commands

### Setup
```bash
# Clone or create in project directory
cd titivillus/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
# Or install with optional dependencies
pip install -e .[dev,plotting,fast]
```

### CLI Usage
```bash
# Generate dataset from configuration
titivillus generate --config config.yaml --output ./results

# Quick dataset generation using templates
titivillus generate --template linguistic --taxa 20 --characters 50 --output ./data

# Validate configuration file
titivillus validate --config config.yaml

# Create example configuration
titivillus init --output example_config.yaml --template linguistic

# List available templates
titivillus templates --list

# Show template details
titivillus templates --show linguistic

# Show program information
titivillus info
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_generator.py

# Run with coverage
pytest --cov=titivillus tests/

# Run only fast tests (skip slow integration tests)
pytest -m "not slow" tests/

# Run specific domain tests
pytest tests/test_domains.py::test_linguistic_generation
```

### Development Tools
```bash
# Format code with black
black titivillus/ tests/

# Check types with mypy  
mypy titivillus/

# Lint with flake8
flake8 titivillus/

# Check setup.py and package
python setup.py check --strict
```

### Building and Distribution
```bash
# Build source and wheel distributions
python -m build

# Check distribution
twine check dist/*

# Upload to Test PyPI (for testing)
twine upload --repository testpypi dist/*

# Upload to PyPI (for release)
twine upload dist/*
```

## Architecture

### Design Principles
- **CLI-First**: Primary interface is command-line with YAML configurations
- **Modular Components**: Clear separation between domains, evolution, networks, errors
- **Ground Truth Tracking**: Complete provenance and validation data for every generated dataset
- **Realistic Modeling**: Embrace complexity and messiness of real phylogenetic data
- **Software Compatibility**: Generate data compatible with PAUP*, MrBayes, and other phylogenetic software

### Module Structure
```
titivillus/
├── __init__.py           # Public API and imports
├── cli.py                # Command-line interface
├── config.py             # YAML configuration system
├── generator.py          # Main Generator class
├── domains.py            # Domain-specific character generators
├── evolution.py          # Evolutionary simulation engine  
├── networks.py           # Network effects and reticulation
├── errors.py             # Error injection and missing data
├── outputs.py            # Multi-format output generation
├── validation.py         # Ground truth and validation
├── tree.py               # Phylogenetic tree generation
└── templates.py          # Configuration templates
```

### Core Data Flow
1. **Configuration**: Load YAML config or use template
2. **Tree Generation**: Create base phylogenetic tree
3. **Network Effects**: Apply reticulation and contact zones  
4. **Character Generation**: Create domain-specific characters
5. **Evolution Simulation**: Simulate character evolution on tree
6. **Error Injection**: Add realistic errors and missing data
7. **Ground Truth**: Track complete generation provenance
8. **Output**: Export in multiple formats (NEXUS, CSV, JSON, etc.)

### Key Data Structures

**Config**: Master configuration loaded from YAML
- Tree parameters (taxa count, branch lengths, calibrations)
- Domain configurations (linguistics, stemmatology, cultural)
- Network parameters (reticulation rates, contact zones)
- Error parameters (missing data, measurement errors)
- Output specifications (formats, options)

**DatasetResult**: Complete generated dataset
- Taxa list and character definitions
- Character data matrix
- Ground truth information (true tree, evolution histories)
- Network information (reticulations, contact zones)
- Generation statistics and metadata

**GroundTruth**: Comprehensive validation data
- True phylogenetic tree and branch lengths
- Character evolution histories and state changes
- Error patterns and missing data locations
- Network effects and reticulation events
- Generation configuration and parameters

## Configuration System

### YAML Structure
```yaml
name: "my_synthetic_dataset"
description: "Custom dataset for testing"
seed: 42

tree:
  taxa_count: 25
  tree_height: 1.0
  tree_shape: "balanced"

active_domains:
  - "linguistics"
  - "stemmatology"

domains:
  linguistics:
    character_count: 60
    substitution_rate: 1.2
    asymmetric_bias: 0.7
    borrowability_range: [0.1, 0.8]
    
  stemmatology:
    character_count: 40
    scribal_error_rate: 0.15
    variant_complexity: 0.8

networks:
  enable_reticulation: true
  reticulation_rate: 0.15
  contact_zones: 3

errors:
  missing_data_rate: 0.20
  measurement_error_rate: 0.08
  polymorphism_rate: 0.12

output:
  formats: ["nexus", "csv", "json"]
  base_name: "synthetic_data"
  include_timestamp: true
```

### Domain-Specific Parameters

**Linguistics**:
- `borrowability_range`: Range for character borrowability [min, max]
- `phonological_constraints`: Enable sound change constraints
- `areal_effects`: Strength of geographic contact effects
- `sound_change_rates`: Rates for different phonological processes

**Stemmatology**:
- `scribal_error_rate`: Base rate of copying errors
- `variant_complexity`: Complexity of textual variants
- `contamination_events`: Cross-contamination between manuscripts
- `paleographic_confusion`: Enable letter confusion patterns

**Cultural Evolution**:
- `innovation_bias`: Tendency toward novel cultural traits
- `diffusion_strength`: Horizontal transmission rate
- `prestige_effects`: Influence of high-status variants
- `trade_route_effects`: Cultural transmission via trade

## Implementation Guidelines

### Adding New Features

**New Domain Support**:
1. Create domain class inheriting from `DomainGenerator`
2. Add domain-specific configuration class
3. Implement character generation methods
4. Add domain to `Domain` enum and factory
5. Update documentation and examples

**New Output Format**:
1. Create writer class inheriting from `OutputWriter`
2. Implement format-specific export methods
3. Add format to `OutputFormat` enum
4. Update output engine and CLI options
5. Add format tests and examples

**New Error Pattern**:
1. Add error type to `ErrorType` enum
2. Implement error injection method in `ErrorEngine`
3. Add configuration parameters
4. Update ground truth tracking
5. Add tests and documentation

### Testing Strategy

**Unit Tests**:
- Test individual components in isolation
- Mock dependencies and random number generation
- Verify configuration validation and error handling
- Test edge cases and boundary conditions

**Integration Tests**:
- Test complete generation pipeline end-to-end
- Verify output format correctness and compatibility
- Test with various configuration combinations
- Validate ground truth accuracy

**Property-Based Tests**:
- Use hypothesis to test with random valid configurations
- Verify invariants across different parameter ranges
- Test edge cases automatically discovered

**Performance Tests**:
- Benchmark generation time for different dataset sizes
- Memory usage profiling for large datasets
- Scalability testing with high taxa/character counts

## Output Formats

### NEXUS Format
- Compatible with PAUP*, MrBayes, and other phylogenetic software
- Includes character labels, state definitions, and assumptions blocks
- Trees block with true topology and branch lengths
- Character sets for partitioned analysis

### CSV Format
- Character data matrix for statistical analysis
- Separate metadata file with character properties
- Optional partition-specific files
- Missing data and polymorphisms properly encoded

### JSON Format
- Complete metadata and provenance information
- Ground truth data for validation
- Nested structure with generation configuration
- Machine-readable for automated analysis pipelines

## Quality Assurance

### Code Style
- Follow PEP 8 Python style guidelines
- Use Black for automatic formatting
- Type hints for all public APIs
- Comprehensive docstrings in Google style

### Documentation Standards
- Clear, concise documentation for all features
- Practical examples for common use cases
- API reference with parameter descriptions
- Architecture explanations for contributors

### Error Handling
- Graceful handling of configuration errors
- Informative error messages with suggestions
- Validation of all user inputs
- Proper logging at appropriate levels

## Phylogenetic Software Compatibility

### PAUP*
- Generate NEXUS files with proper character encodings
- Include assumptions blocks for model specifications
- Support for ordered/unordered character types
- Compatible tree format with branch lengths

### MrBayes
- Proper NEXUS format with model blocks
- Support for partitioned analysis
- Character state space definitions
- Compatible prior specifications

### General Phylogenetic Software
- Standard Newick tree format
- FASTA sequence format support
- CSV matrices for R/Python analysis
- Standardized missing data encoding

## Performance Considerations

### Memory Management
- Stream processing for large datasets where possible
- Lazy evaluation of expensive computations
- Proper cleanup of temporary data structures
- Memory profiling for optimization

### Computational Efficiency
- NumPy arrays for numerical computations
- Optional Numba acceleration for hot paths
- Efficient algorithms for tree operations
- Parallel processing where appropriate

### Scalability
- Linear scaling with number of characters
- Reasonable performance up to 1000+ taxa
- Configurable precision vs. performance trade-offs
- Progress reporting for long-running generations

## Contributing Guidelines

### Development Workflow
1. Fork repository and create feature branch
2. Write tests for new functionality
3. Implement feature with proper documentation
4. Run full test suite and quality checks
5. Submit pull request with clear description

### Code Review Process
- All changes require review by maintainer
- Tests must pass on all supported Python versions
- Documentation must be updated for new features
- Performance impact should be considered

### Release Process
- Semantic versioning (MAJOR.MINOR.PATCH)
- Changelog with clear feature descriptions
- GitHub releases with distribution files
- PyPI package updates

## Important Notes for Claude Code

### When Working with This Codebase
- **Respect the CLI-first design**: Most functionality should be accessible via command line
- **Maintain YAML compatibility**: Configuration changes must preserve backward compatibility
- **Include comprehensive ground truth**: Any new features must properly track provenance
- **Test with real phylogenetic software**: Verify output compatibility with PAUP*, MrBayes
- **Follow domain expertise**: Ensure biological/linguistic realism in modeling decisions
- **Performance matters**: Consider scalability for large datasets (1000+ taxa)

### Code Quality Standards
- All public functions must have type hints and docstrings
- Configuration validation with helpful error messages
- Comprehensive test coverage including edge cases
- Memory-efficient implementation for large datasets

### Documentation Requirements
- Update CLAUDE.md for any architectural changes
- Include practical examples in docstrings
- Update CLI help text for new commands/options
- Maintain compatibility notes for phylogenetic software
