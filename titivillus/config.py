"""
Configuration system for Titivillus synthetic data generation.

This module provides a comprehensive YAML-based configuration system that allows
users to specify all aspects of synthetic phylogenetic data generation through
external configuration files.
"""

import yaml
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Domain(Enum):
    """Supported phylogenetic domains."""
    LINGUISTICS = "linguistics"
    STEMMATOLOGY = "stemmatology" 
    CULTURAL = "cultural"


class SimulationMode(Enum):
    """Evolutionary simulation approaches."""
    CTMC = "continuous_time_markov_chain"
    DISCRETE = "discrete_events"
    HYBRID = "hybrid"


class OutputFormat(Enum):
    """Supported output formats."""
    NEXUS = "nexus"
    NEWICK = "newick"
    CSV = "csv"
    JSON = "json"
    FASTA = "fasta"


class ErrorType(Enum):
    """Types of data errors to inject."""
    MISSING = "missing"
    MEASUREMENT = "measurement_error"
    POLYMORPHISM = "polymorphism"
    CONTAMINATION = "contamination"


class MissingPattern(Enum):
    """Patterns for missing data injection."""
    RANDOM = "random"
    SYSTEMATIC = "systematic"
    CLUSTERED = "clustered"
    TAXON_SPECIFIC = "taxon_specific"


@dataclass
class DomainConfig:
    """Configuration for domain-specific generation."""
    # Character generation
    character_count: int = 50
    character_types: List[str] = field(default_factory=lambda: ["binary", "multistate"])
    
    # Evolution parameters
    substitution_rate: float = 1.0
    asymmetric_bias: float = 0.0  # 0.0 = symmetric, 1.0 = fully asymmetric
    directional_change_prob: float = 0.0
    
    # Domain-specific parameters (will be overridden by specific domains)
    domain_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinguisticConfig(DomainConfig):
    """Linguistic-specific configuration parameters."""
    def __post_init__(self):
        # Linguistic-specific defaults
        self.domain_params.update({
            'borrowability_range': [0.1, 0.8],
            'phonological_constraints': True,
            'morphological_complexity': 0.6,
            'lexical_stability': 0.8,
            'areal_effects': 0.3,
            'sound_change_rates': {
                'consonantal': 0.8,
                'vocalic': 1.2,
                'prosodic': 0.6
            }
        })


@dataclass  
class StemmatologyConfig(DomainConfig):
    """Stemmatology-specific configuration parameters."""
    def __post_init__(self):
        # Stemmatology-specific defaults
        self.domain_params.update({
            'scribal_error_rate': 0.15,
            'variant_complexity': 0.7,
            'manuscript_quality_variation': 0.4,
            'contamination_events': 0.1,
            'error_prone_positions': 0.25,
            'paleographic_confusion': True,
            'correction_probability': 0.3
        })


@dataclass
class CulturalConfig(DomainConfig):
    """Cultural evolution-specific configuration parameters."""
    def __post_init__(self):
        # Cultural evolution-specific defaults  
        self.domain_params.update({
            'innovation_bias': 0.4,
            'diffusion_strength': 0.6,
            'prestige_effects': 0.3,
            'trade_route_effects': 0.2,
            'stability_variation': 0.4,
            'horizontal_transmission': 0.5,
            'vertical_transmission': 0.8
        })


@dataclass
class ErrorConfig:
    """Configuration for error injection."""
    # Missing data
    missing_data_rate: float = 0.15
    missing_patterns: List[MissingPattern] = field(
        default_factory=lambda: [MissingPattern.RANDOM, MissingPattern.SYSTEMATIC]
    )
    systematic_missing_bias: float = 0.3
    
    # Measurement errors
    measurement_error_rate: float = 0.05
    error_clustering: bool = False
    cluster_size_range: List[int] = field(default_factory=lambda: [2, 5])
    
    # Polymorphisms
    polymorphism_rate: float = 0.08
    polymorphism_complexity: float = 0.6
    
    # Domain-specific error patterns
    domain_specific_errors: bool = True


@dataclass
class NetworkConfig:
    """Configuration for network effects and reticulation."""
    # Reticulation parameters
    enable_reticulation: bool = True
    reticulation_rate: float = 0.1
    max_reticulations: int = 5
    reticulation_strength: float = 0.3
    
    # Contact zones
    contact_zones: int = 2
    contact_strength: float = 0.4
    contact_duration_range: List[float] = field(default_factory=lambda: [0.1, 0.5])
    
    # Borrowing and horizontal transfer
    borrowing_rate: float = 0.2
    horizontal_transfer_rate: float = 0.15
    
    # Geographic structure
    geographic_structure: bool = True
    migration_rate: float = 0.05


@dataclass
class TreeConfig:
    """Configuration for phylogenetic tree generation."""
    # Tree structure
    taxa_count: int = 20
    tree_height: float = 1.0
    tree_shape: str = "balanced"  # balanced, pectinate, random
    
    # Branch lengths
    branch_length_distribution: str = "exponential"  # exponential, uniform, gamma
    branch_length_params: Dict[str, float] = field(default_factory=lambda: {"rate": 1.0})
    
    # Root and calibration
    root_age: Optional[float] = None
    calibration_points: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OutputConfig:
    """Configuration for output generation."""
    formats: List[OutputFormat] = field(
        default_factory=lambda: [OutputFormat.NEXUS, OutputFormat.CSV, OutputFormat.JSON]
    )
    
    # File naming
    base_name: str = "synthetic_dataset"
    include_timestamp: bool = True
    
    # Format-specific options
    nexus_options: Dict[str, Any] = field(default_factory=lambda: {
        "include_character_labels": True,
        "include_state_labels": True,
        "include_assumptions": True,
        "include_trees": True
    })
    
    csv_options: Dict[str, Any] = field(default_factory=lambda: {
        "separate_partitions": True,
        "include_metadata": True
    })
    
    json_options: Dict[str, Any] = field(default_factory=lambda: {
        "include_ground_truth": True,
        "compact": False
    })


@dataclass
class Config:
    """Master configuration for synthetic data generation."""
    # Basic parameters
    name: str = "synthetic_phylogenetic_dataset"
    description: str = ""
    seed: int = 42
    
    # Tree configuration
    tree: TreeConfig = field(default_factory=TreeConfig)
    
    # Domain configurations
    domains: Dict[Domain, DomainConfig] = field(default_factory=lambda: {
        Domain.LINGUISTICS: LinguisticConfig(),
        Domain.STEMMATOLOGY: StemmatologyConfig(),
        Domain.CULTURAL: CulturalConfig()
    })
    
    # Active domains (subset of above)
    active_domains: List[Domain] = field(
        default_factory=lambda: [Domain.LINGUISTICS, Domain.STEMMATOLOGY]
    )
    
    # Error and network effects
    errors: ErrorConfig = field(default_factory=ErrorConfig)
    networks: NetworkConfig = field(default_factory=NetworkConfig)
    
    # Simulation parameters
    simulation_mode: SimulationMode = SimulationMode.HYBRID
    
    # Output configuration
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Validation options
    validation_level: str = "comprehensive"  # basic, standard, comprehensive
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
            
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            
        return cls._from_dict(yaml_data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Internal method to create Config from dictionary with validation."""
        # Start with defaults
        config = cls()
        
        # Update basic parameters
        if 'name' in data:
            config.name = data['name']
        if 'description' in data:
            config.description = data['description']
        if 'seed' in data:
            config.seed = int(data['seed'])
        if 'simulation_mode' in data:
            config.simulation_mode = SimulationMode(data['simulation_mode'])
        if 'validation_level' in data:
            config.validation_level = data['validation_level']
            
        # Update tree configuration
        if 'tree' in data:
            tree_data = data['tree']
            config.tree = TreeConfig(
                taxa_count=tree_data.get('taxa_count', config.tree.taxa_count),
                tree_height=tree_data.get('tree_height', config.tree.tree_height),
                tree_shape=tree_data.get('tree_shape', config.tree.tree_shape),
                branch_length_distribution=tree_data.get(
                    'branch_length_distribution', 
                    config.tree.branch_length_distribution
                ),
                branch_length_params=tree_data.get(
                    'branch_length_params', 
                    config.tree.branch_length_params
                ),
                root_age=tree_data.get('root_age'),
                calibration_points=tree_data.get('calibration_points', [])
            )
        
        # Update domain configurations
        if 'domains' in data:
            config._update_domains(data['domains'])
            
        if 'active_domains' in data:
            config.active_domains = [Domain(d) for d in data['active_domains']]
            
        # Update error configuration
        if 'errors' in data:
            error_data = data['errors']
            config.errors = ErrorConfig(
                missing_data_rate=error_data.get(
                    'missing_data_rate', config.errors.missing_data_rate
                ),
                missing_patterns=[
                    MissingPattern(p) for p in error_data.get(
                        'missing_patterns', [p.value for p in config.errors.missing_patterns]
                    )
                ],
                systematic_missing_bias=error_data.get(
                    'systematic_missing_bias', config.errors.systematic_missing_bias
                ),
                measurement_error_rate=error_data.get(
                    'measurement_error_rate', config.errors.measurement_error_rate
                ),
                error_clustering=error_data.get(
                    'error_clustering', config.errors.error_clustering
                ),
                cluster_size_range=error_data.get(
                    'cluster_size_range', config.errors.cluster_size_range
                ),
                polymorphism_rate=error_data.get(
                    'polymorphism_rate', config.errors.polymorphism_rate
                ),
                polymorphism_complexity=error_data.get(
                    'polymorphism_complexity', config.errors.polymorphism_complexity
                ),
                domain_specific_errors=error_data.get(
                    'domain_specific_errors', config.errors.domain_specific_errors
                )
            )
            
        # Update network configuration
        if 'networks' in data:
            network_data = data['networks']
            config.networks = NetworkConfig(
                enable_reticulation=network_data.get(
                    'enable_reticulation', config.networks.enable_reticulation
                ),
                reticulation_rate=network_data.get(
                    'reticulation_rate', config.networks.reticulation_rate
                ),
                max_reticulations=network_data.get(
                    'max_reticulations', config.networks.max_reticulations
                ),
                reticulation_strength=network_data.get(
                    'reticulation_strength', config.networks.reticulation_strength
                ),
                contact_zones=network_data.get(
                    'contact_zones', config.networks.contact_zones
                ),
                contact_strength=network_data.get(
                    'contact_strength', config.networks.contact_strength
                ),
                contact_duration_range=network_data.get(
                    'contact_duration_range', config.networks.contact_duration_range
                ),
                borrowing_rate=network_data.get(
                    'borrowing_rate', config.networks.borrowing_rate
                ),
                horizontal_transfer_rate=network_data.get(
                    'horizontal_transfer_rate', config.networks.horizontal_transfer_rate
                ),
                geographic_structure=network_data.get(
                    'geographic_structure', config.networks.geographic_structure
                ),
                migration_rate=network_data.get(
                    'migration_rate', config.networks.migration_rate
                )
            )
            
        # Update output configuration
        if 'output' in data:
            output_data = data['output']
            config.output = OutputConfig(
                formats=[OutputFormat(f) for f in output_data.get(
                    'formats', [f.value for f in config.output.formats]
                )],
                base_name=output_data.get('base_name', config.output.base_name),
                include_timestamp=output_data.get(
                    'include_timestamp', config.output.include_timestamp
                ),
                nexus_options=output_data.get('nexus_options', config.output.nexus_options),
                csv_options=output_data.get('csv_options', config.output.csv_options),
                json_options=output_data.get('json_options', config.output.json_options)
            )
            
        return config
    
    def _update_domains(self, domains_data: Dict[str, Any]) -> None:
        """Update domain configurations from dictionary."""
        for domain_name, domain_data in domains_data.items():
            try:
                domain = Domain(domain_name)
            except ValueError:
                logger.warning(f"Unknown domain: {domain_name}")
                continue
                
            # Create appropriate domain config based on type
            if domain == Domain.LINGUISTICS:
                domain_config = LinguisticConfig()
            elif domain == Domain.STEMMATOLOGY:
                domain_config = StemmatologyConfig()
            elif domain == Domain.CULTURAL:
                domain_config = CulturalConfig()
            else:
                domain_config = DomainConfig()
                
            # Update with provided values
            for key, value in domain_data.items():
                if hasattr(domain_config, key):
                    setattr(domain_config, key, value)
                else:
                    # Add to domain_params
                    domain_config.domain_params[key] = value
                    
            self.domains[domain] = domain_config
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary for YAML serialization
        config_dict = self._to_dict()
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'seed': self.seed,
            'simulation_mode': self.simulation_mode.value,
            'validation_level': self.validation_level,
            
            'tree': {
                'taxa_count': self.tree.taxa_count,
                'tree_height': self.tree.tree_height,
                'tree_shape': self.tree.tree_shape,
                'branch_length_distribution': self.tree.branch_length_distribution,
                'branch_length_params': self.tree.branch_length_params,
                'root_age': self.tree.root_age,
                'calibration_points': self.tree.calibration_points
            },
            
            'active_domains': [d.value for d in self.active_domains],
            
            'domains': {
                domain.value: {
                    'character_count': config.character_count,
                    'character_types': config.character_types,
                    'substitution_rate': config.substitution_rate,
                    'asymmetric_bias': config.asymmetric_bias,
                    'directional_change_prob': config.directional_change_prob,
                    **config.domain_params
                }
                for domain, config in self.domains.items()
            },
            
            'errors': {
                'missing_data_rate': self.errors.missing_data_rate,
                'missing_patterns': [p.value for p in self.errors.missing_patterns],
                'systematic_missing_bias': self.errors.systematic_missing_bias,
                'measurement_error_rate': self.errors.measurement_error_rate,
                'error_clustering': self.errors.error_clustering,
                'cluster_size_range': self.errors.cluster_size_range,
                'polymorphism_rate': self.errors.polymorphism_rate,
                'polymorphism_complexity': self.errors.polymorphism_complexity,
                'domain_specific_errors': self.errors.domain_specific_errors
            },
            
            'networks': {
                'enable_reticulation': self.networks.enable_reticulation,
                'reticulation_rate': self.networks.reticulation_rate,
                'max_reticulations': self.networks.max_reticulations,
                'reticulation_strength': self.networks.reticulation_strength,
                'contact_zones': self.networks.contact_zones,
                'contact_strength': self.networks.contact_strength,
                'contact_duration_range': self.networks.contact_duration_range,
                'borrowing_rate': self.networks.borrowing_rate,
                'horizontal_transfer_rate': self.networks.horizontal_transfer_rate,
                'geographic_structure': self.networks.geographic_structure,
                'migration_rate': self.networks.migration_rate
            },
            
            'output': {
                'formats': [f.value for f in self.output.formats],
                'base_name': self.output.base_name,
                'include_timestamp': self.output.include_timestamp,
                'nexus_options': self.output.nexus_options,
                'csv_options': self.output.csv_options,
                'json_options': self.output.json_options
            }
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Basic validation
        if self.tree.taxa_count < 3:
            issues.append("taxa_count must be at least 3")
            
        if not (0.0 <= self.errors.missing_data_rate <= 1.0):
            issues.append("missing_data_rate must be between 0.0 and 1.0")
            
        if not (0.0 <= self.networks.reticulation_rate <= 1.0):
            issues.append("reticulation_rate must be between 0.0 and 1.0")
            
        # Domain validation
        for domain in self.active_domains:
            if domain not in self.domains:
                issues.append(f"Active domain {domain.value} not configured")
                continue
                
            domain_config = self.domains[domain]
            if domain_config.character_count < 1:
                issues.append(f"Domain {domain.value}: character_count must be at least 1")
                
        return issues