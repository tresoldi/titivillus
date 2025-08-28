"""
Titivillus: Synthetic Phylogenetic Data Generator
================================================

Titivillus (named after the demon of scribal errors) is a comprehensive library 
for generating synthetic phylogenetic datasets with realistic evolutionary 
processes, network effects, and data quality issues. Designed specifically 
for testing and validating phylogenetic inference algorithms.

Core Features:
- Multi-domain support: Linguistics, stemmatology, cultural evolution
- Realistic evolutionary modeling with asymmetric step matrices
- Network effects: Reticulation, borrowing, contact zones
- Comprehensive error injection: Missing data, measurement errors, polymorphisms
- Ground truth tracking for algorithm validation
- Multiple output formats: NEXUS, Newick, CSV, JSON
- YAML-based configuration system
- CLI-first design for easy integration into workflows

Example Usage:
    from titivillus import Generator, Config
    
    # Create generator from YAML config
    config = Config.from_yaml('my_config.yaml')
    generator = Generator(config)
    
    # Generate synthetic dataset
    dataset = generator.generate()
    
    # Export in multiple formats
    generator.export(dataset, output_dir='./results')

Command Line Usage:
    # Generate dataset from config
    titivillus generate --config config.yaml --output ./results
    
    # Validate configuration
    titivillus validate --config config.yaml
    
    # List available templates
    titivillus templates

Philosophy:
Titivillus embraces the inherent messiness of real phylogenetic data while 
providing complete ground truth for rigorous algorithm testing. Named after 
the medieval demon who introduced errors into manuscripts, it acknowledges 
that realistic synthetic data must include the imperfections that make 
phylogenetic inference challenging.
"""

__version__ = '0.1.0'
__author__ = 'Titivillus Development Team'
__license__ = 'MIT'

# Core imports for public API - only import what exists
from .config import Config
from .generator import Generator
from .domains import Domain
from .tree import TreeGenerator

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',
    
    # Core classes
    'Generator',
    'Config',
    'Domain',
    'TreeGenerator',
]