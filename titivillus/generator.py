"""
Main Generator class for Titivillus synthetic phylogenetic data generation.

This module provides the central Generator class that coordinates all aspects
of synthetic data generation including evolution simulation, network effects,
error injection, and output generation.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime

from .config import Config, Domain
from .domains import DomainFactory
from .evolution import EvolutionEngine
from .networks import NetworkEngine
from .errors import ErrorEngine
from .outputs import OutputEngine
from .validation import GroundTruth
from .tree import TreeGenerator

logger = logging.getLogger(__name__)


@dataclass
class DatasetResult:
    """Result container for generated synthetic dataset."""
    name: str
    config: Config
    generation_time: datetime
    
    # Core data
    taxa: List[str]
    characters: List[Dict[str, Any]]
    character_data: Dict[str, Dict[int, Any]]
    
    # Ground truth information
    ground_truth: GroundTruth
    
    # Tree information
    tree_newick: str
    branch_lengths: Dict[str, float]
    
    # Network information (if applicable)
    reticulations: List[Dict[str, Any]] = field(default_factory=list)
    contact_zones: List[Dict[str, Any]] = field(default_factory=list)
    
    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the dataset."""
        return {
            'name': self.name,
            'generation_time': self.generation_time.isoformat(),
            'taxa_count': len(self.taxa),
            'character_count': len(self.characters),
            'domains': list(set(char.domain for char in self.characters)),
            'reticulations': len(self.reticulations),
            'contact_zones': len(self.contact_zones),
            'missing_data_rate': self._calculate_missing_rate(),
            'polymorphism_rate': self._calculate_polymorphism_rate(),
            'stats': self.stats
        }
    
    def _calculate_missing_rate(self) -> float:
        """Calculate actual missing data rate in the dataset."""
        total_cells = len(self.taxa) * len(self.characters)
        missing_cells = 0
        
        for taxon_data in self.character_data.values():
            for state in taxon_data.values():
                if state is None or state == '?':
                    missing_cells += 1
                    
        return missing_cells / total_cells if total_cells > 0 else 0.0
    
    def _calculate_polymorphism_rate(self) -> float:
        """Calculate actual polymorphism rate in the dataset."""
        total_cells = len(self.taxa) * len(self.characters)
        polymorphic_cells = 0
        
        for taxon_data in self.character_data.values():
            for state in taxon_data.values():
                if isinstance(state, str) and ('/' in state or ',' in state):
                    polymorphic_cells += 1
                    
        return polymorphic_cells / total_cells if total_cells > 0 else 0.0


class Generator:
    """
    Main generator for synthetic phylogenetic datasets.
    
    The Generator coordinates all aspects of synthetic data creation:
    - Phylogenetic tree generation
    - Domain-specific character evolution
    - Network effects and reticulation
    - Error injection and data degradation
    - Ground truth tracking
    - Multi-format output generation
    """
    
    def __init__(self, config: Config):
        """Initialize generator with configuration."""
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
        # Initialize component engines
        self._setup_engines()
        
        logger.info(f"Initialized Titivillus generator with seed {config.seed}")
    
    def _setup_engines(self) -> None:
        """Initialize all component engines."""
        # Core engines
        self.tree_generator = TreeGenerator(self.config.tree, seed=self.config.seed)
        self.evolution_engine = EvolutionEngine(
            mode=self.config.simulation_mode, 
            seed=self.config.seed
        )
        self.network_engine = NetworkEngine(self.config.networks, seed=self.config.seed)
        self.error_engine = ErrorEngine(self.config.errors, seed=self.config.seed)
        self.output_engine = OutputEngine(self.config.output)
        
        # Domain factories
        self.domain_factory = DomainFactory(seed=self.config.seed)
        
        logger.debug("Component engines initialized")
    
    def generate(self, name: Optional[str] = None) -> DatasetResult:
        """
        Generate a complete synthetic phylogenetic dataset.
        
        Args:
            name: Optional name for the dataset (uses config name if not provided)
            
        Returns:
            DatasetResult containing all generated data and ground truth
        """
        generation_start = datetime.now()
        dataset_name = name or self.config.name
        
        logger.info(f"Generating synthetic dataset: {dataset_name}")
        
        # Step 1: Generate base phylogenetic tree
        logger.debug("Generating base phylogenetic tree")
        tree_result = self.tree_generator.generate()
        taxa = tree_result.taxa
        tree_newick = tree_result.newick
        branch_lengths = tree_result.branch_lengths
        
        # Step 2: Apply network effects if enabled
        reticulations = []
        contact_zones = []
        if self.config.networks.enable_reticulation:
            logger.debug("Applying network effects")
            network_result = self.network_engine.apply_network_effects(
                tree_result, taxa
            )
            tree_newick = network_result['modified_tree']
            reticulations = network_result.get('reticulations', [])
            contact_zones = network_result.get('contact_zones', [])
        
        # Step 3: Generate domain-specific characters
        logger.debug("Generating domain-specific characters")
        all_characters = []
        for domain in self.config.active_domains:
            domain_config = self.config.domains[domain]
            domain_generator = self.domain_factory.get_generator(domain, domain_config)
            
            characters = domain_generator.generate_characters(
                count=domain_config.character_count
            )
            
            # Characters already have domain set in domain generator
                
            all_characters.extend(characters)
        
        # Step 4: Simulate character evolution
        logger.debug("Simulating character evolution")
        character_data = {}
        for taxon in taxa:
            character_data[taxon] = {}
            
        evolution_result = self.evolution_engine.evolve_characters(
            characters=all_characters,
            tree_info={
                'taxa': taxa,
                'newick': tree_newick,
                'branch_lengths': branch_lengths,
                'reticulations': reticulations
            }
        )
        
        # Update character data with evolution results
        character_data.update(evolution_result['character_data'])
        character_histories = evolution_result.get('histories', {})
        
        # Step 4.5: Apply network-based character borrowing
        borrowing_events = []
        if reticulations or contact_zones:
            logger.debug("Applying network-based character borrowing")
            character_data, borrowing_events = self.network_engine.apply_character_borrowing(
                character_data, all_characters, taxa, reticulations, contact_zones
            )
        
        # Step 5: Apply error injection
        logger.debug("Applying error injection")
        degraded_data, error_patterns = self.error_engine.inject_errors(
            character_data, all_characters, taxa
        )
        character_data = degraded_data
        
        # Step 6: Create ground truth object
        logger.debug("Creating ground truth tracking")
        ground_truth = GroundTruth(
            true_tree=tree_newick,
            branch_lengths=branch_lengths,
            character_histories=character_histories,
            error_patterns=error_patterns,
            reticulations=reticulations,
            contact_zones=contact_zones,
            borrowing_events=borrowing_events,
            generation_config=self.config,
            validation_level=self.config.validation_level
        )
        
        # Step 7: Calculate statistics
        stats = self._calculate_generation_stats(
            taxa, all_characters, character_data, 
            reticulations, contact_zones, generation_start
        )
        
        # Create final dataset result
        result = DatasetResult(
            name=dataset_name,
            config=self.config,
            generation_time=generation_start,
            taxa=taxa,
            characters=all_characters,
            character_data=character_data,
            ground_truth=ground_truth,
            tree_newick=tree_newick,
            branch_lengths=branch_lengths,
            reticulations=reticulations,
            contact_zones=contact_zones,
            stats=stats
        )
        
        logger.info(f"Generated dataset '{dataset_name}' with {len(taxa)} taxa, "
                   f"{len(all_characters)} characters in {stats['generation_time']:.2f}s")
        
        return result
    
    def export(self, dataset: DatasetResult, output_dir: str) -> Dict[str, Path]:
        """
        Export dataset in configured output formats.
        
        Args:
            dataset: Generated dataset to export
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping format names to output file paths
        """
        logger.info(f"Exporting dataset to {output_dir}")
        
        output_paths = self.output_engine.export_dataset(
            dataset=dataset,
            output_dir=Path(output_dir)
        )
        
        logger.info(f"Exported {len(output_paths)} files: {list(output_paths.keys())}")
        return output_paths
    
    def generate_and_export(self, output_dir: str, name: Optional[str] = None) -> Tuple[DatasetResult, Dict[str, Path]]:
        """
        Convenience method to generate dataset and export in one step.
        
        Args:
            output_dir: Directory for output files
            name: Optional name for dataset
            
        Returns:
            Tuple of (dataset_result, output_paths)
        """
        dataset = self.generate(name)
        output_paths = self.export(dataset, output_dir)
        return dataset, output_paths
    
    def _calculate_generation_stats(
        self, 
        taxa: List[str], 
        characters: List[Dict[str, Any]], 
        character_data: Dict[str, Dict[int, Any]],
        reticulations: List[Dict[str, Any]],
        contact_zones: List[Dict[str, Any]],
        start_time: datetime
    ) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the generated dataset."""
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Domain distribution
        domain_counts = {}
        for char in characters:
            domain = char.domain if hasattr(char, 'domain') else 'unknown'
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Character type distribution
        char_type_counts = {}
        for char in characters:
            char_type = char.character_type.value if hasattr(char, 'character_type') else 'unknown'
            char_type_counts[char_type] = char_type_counts.get(char_type, 0) + 1
        
        # Data matrix statistics
        total_cells = len(taxa) * len(characters)
        missing_cells = 0
        polymorphic_cells = 0
        
        for taxon_data in character_data.values():
            for state in taxon_data.values():
                if state is None or state == '?':
                    missing_cells += 1
                elif isinstance(state, str) and ('/' in state or ',' in state):
                    polymorphic_cells += 1
        
        return {
            'generation_time': generation_time,
            'taxa_count': len(taxa),
            'character_count': len(characters),
            'total_data_cells': total_cells,
            'missing_cells': missing_cells,
            'missing_rate': missing_cells / total_cells if total_cells > 0 else 0,
            'polymorphic_cells': polymorphic_cells,
            'polymorphism_rate': polymorphic_cells / total_cells if total_cells > 0 else 0,
            'domain_distribution': domain_counts,
            'character_type_distribution': char_type_counts,
            'reticulation_count': len(reticulations),
            'contact_zone_count': len(contact_zones),
            'network_effects_enabled': len(reticulations) > 0 or len(contact_zones) > 0
        }
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        Validate the current configuration.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = self.config.validate()
        
        # Additional validation specific to generation
        if not self.config.active_domains:
            issues.append("At least one domain must be active")
            
        # Check domain configurations
        for domain in self.config.active_domains:
            if domain not in self.config.domains:
                issues.append(f"Active domain {domain.value} is not configured")
        
        return len(issues) == 0, issues
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Generator':
        """
        Create generator from YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configured Generator instance
        """
        config = Config.from_yaml(config_path)
        return cls(config)
    
    @classmethod
    def quick_linguistic(cls, taxa_count: int = 20, character_count: int = 50, 
                        seed: int = 42) -> 'Generator':
        """
        Create a generator for quick linguistic dataset generation.
        
        Args:
            taxa_count: Number of taxa
            character_count: Number of characters
            seed: Random seed
            
        Returns:
            Configured Generator for linguistic data
        """
        config = Config(
            name=f"quick_linguistic_{taxa_count}taxa_{character_count}chars",
            seed=seed,
            active_domains=[Domain.LINGUISTICS]
        )
        config.tree.taxa_count = taxa_count
        config.domains[Domain.LINGUISTICS].character_count = character_count
        
        return cls(config)
    
    @classmethod  
    def quick_stemmatology(cls, taxa_count: int = 15, character_count: int = 40,
                          seed: int = 42) -> 'Generator':
        """
        Create a generator for quick stemmatology dataset generation.
        
        Args:
            taxa_count: Number of manuscripts
            character_count: Number of variant sites
            seed: Random seed
            
        Returns:
            Configured Generator for stemmatology data
        """
        config = Config(
            name=f"quick_stemmatology_{taxa_count}mss_{character_count}vars",
            seed=seed,
            active_domains=[Domain.STEMMATOLOGY]
        )
        config.tree.taxa_count = taxa_count
        config.domains[Domain.STEMMATOLOGY].character_count = character_count
        
        return cls(config)
    
    @classmethod
    def quick_multidomain(cls, taxa_count: int = 25, seed: int = 42) -> 'Generator':
        """
        Create a generator for quick multi-domain dataset generation.
        
        Args:
            taxa_count: Number of taxa
            seed: Random seed
            
        Returns:
            Configured Generator for multi-domain data
        """
        config = Config(
            name=f"quick_multidomain_{taxa_count}taxa",
            seed=seed,
            active_domains=[Domain.LINGUISTICS, Domain.STEMMATOLOGY, Domain.CULTURAL]
        )
        config.tree.taxa_count = taxa_count
        
        # Balanced character counts across domains
        config.domains[Domain.LINGUISTICS].character_count = 40
        config.domains[Domain.STEMMATOLOGY].character_count = 30
        config.domains[Domain.CULTURAL].character_count = 30
        
        return cls(config)