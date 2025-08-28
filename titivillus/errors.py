"""
Error injection engine for Titivillus synthetic phylogenetic data.

This module implements realistic error patterns found in phylogenetic datasets:
- Missing data with various patterns (random, systematic, clustered, taxon-specific)
- Measurement errors and state confusion
- Polymorphisms and ambiguous observations
- Domain-specific error patterns reflecting real-world data collection challenges
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from copy import deepcopy
import random

from .config import MissingPattern, ErrorType

logger = logging.getLogger(__name__)


@dataclass
class ErrorRecord:
    """Record of a specific error injection event."""
    error_type: str
    taxon: str
    character_index: int
    original_state: str
    modified_state: str
    pattern: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorPatterns:
    """Complete record of all error patterns applied to dataset."""
    missing_data: List[ErrorRecord] = field(default_factory=list)
    measurement_errors: List[ErrorRecord] = field(default_factory=list)
    polymorphisms: List[ErrorRecord] = field(default_factory=list)
    
    # Summary statistics
    total_errors: int = 0
    missing_rate_actual: float = 0.0
    error_rate_actual: float = 0.0
    polymorphism_rate_actual: float = 0.0
    
    def add_error(self, record: ErrorRecord) -> None:
        """Add an error record to appropriate category."""
        if record.error_type == "missing":
            self.missing_data.append(record)
        elif record.error_type == "measurement":
            self.measurement_errors.append(record)
        elif record.error_type == "polymorphism":
            self.polymorphisms.append(record)
        
        self.total_errors += 1


class MissingDataGenerator:
    """Generator for missing data patterns."""
    
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
    
    def generate_random_missing(
        self, 
        character_data: Dict[str, Dict[int, Any]], 
        taxa: List[str], 
        characters: List[Any],
        missing_rate: float
    ) -> Set[Tuple[str, int]]:
        """Generate random missing data pattern."""
        
        total_cells = len(taxa) * len(characters)
        n_missing = int(total_cells * missing_rate)
        
        # Randomly select cells to make missing
        missing_cells = set()
        attempts = 0
        max_attempts = n_missing * 3
        
        while len(missing_cells) < n_missing and attempts < max_attempts:
            taxon = self.rng.choice(taxa)
            char_idx = self.rng.randint(0, len(characters))
            missing_cells.add((taxon, char_idx))
            attempts += 1
        
        return missing_cells
    
    def generate_systematic_missing(
        self,
        character_data: Dict[str, Dict[int, Any]], 
        taxa: List[str], 
        characters: List[Any],
        missing_rate: float,
        bias: float = 0.3
    ) -> Set[Tuple[str, int]]:
        """Generate systematic missing data (certain characters/taxa more prone)."""
        
        missing_cells = set()
        
        # Some characters are more prone to missing data
        char_missing_probs = []
        for char in characters:
            base_prob = missing_rate
            # Characters with high missing propensity are more likely to be missing
            if hasattr(char, 'missing_propensity'):
                base_prob += char.missing_propensity * bias
            char_missing_probs.append(min(base_prob * 2, 0.8))  # Cap at 80%
        
        # Some taxa might have systematic issues
        taxon_missing_probs = []
        for taxon in taxa:
            # Add some random taxon-specific bias
            taxon_bias = self.rng.uniform(0.5, 1.5)
            taxon_missing_probs.append(missing_rate * taxon_bias)
        
        # Apply missing data based on combined probabilities
        for i, taxon in enumerate(taxa):
            for j, char in enumerate(characters):
                combined_prob = (char_missing_probs[j] + taxon_missing_probs[i]) / 2
                if self.rng.random() < combined_prob:
                    missing_cells.add((taxon, j))
        
        return missing_cells
    
    def generate_clustered_missing(
        self,
        character_data: Dict[str, Dict[int, Any]], 
        taxa: List[str], 
        characters: List[Any],
        missing_rate: float,
        cluster_size_range: List[int] = [2, 5]
    ) -> Set[Tuple[str, int]]:
        """Generate clustered missing data (missing data occurs in blocks)."""
        
        missing_cells = set()
        total_cells = len(taxa) * len(characters)
        target_missing = int(total_cells * missing_rate)
        
        while len(missing_cells) < target_missing:
            # Choose cluster center
            center_taxon = self.rng.choice(taxa)
            center_char = self.rng.randint(0, len(characters))
            
            # Choose cluster size
            cluster_size = self.rng.randint(*cluster_size_range)
            
            # Generate cluster around center
            for _ in range(cluster_size):
                # Add some randomness to cluster shape
                taxon_offset = self.rng.randint(-1, 2)  # -1, 0, 1
                char_offset = self.rng.randint(-2, 3)   # -2, -1, 0, 1, 2
                
                try:
                    taxon_idx = taxa.index(center_taxon) + taxon_offset
                    char_idx = center_char + char_offset
                    
                    if 0 <= taxon_idx < len(taxa) and 0 <= char_idx < len(characters):
                        missing_cells.add((taxa[taxon_idx], char_idx))
                        
                except (ValueError, IndexError):
                    continue
                
                if len(missing_cells) >= target_missing:
                    break
        
        return missing_cells
    
    def generate_taxon_specific_missing(
        self,
        character_data: Dict[str, Dict[int, Any]], 
        taxa: List[str], 
        characters: List[Any],
        missing_rate: float
    ) -> Set[Tuple[str, int]]:
        """Generate taxon-specific missing patterns (some taxa have more missing data)."""
        
        missing_cells = set()
        
        # Some taxa are "problematic" and have much higher missing rates
        n_problematic = max(1, len(taxa) // 4)  # 25% of taxa are problematic
        problematic_taxa = self.rng.choice(taxa, n_problematic, replace=False)
        
        for taxon in taxa:
            if taxon in problematic_taxa:
                # Problematic taxa have 3x higher missing rate
                taxon_missing_rate = min(missing_rate * 3, 0.6)
            else:
                # Normal taxa have lower missing rate to compensate
                taxon_missing_rate = missing_rate * 0.5
            
            # Apply missing data for this taxon
            for char_idx in range(len(characters)):
                if self.rng.random() < taxon_missing_rate:
                    missing_cells.add((taxon, char_idx))
        
        return missing_cells


class MeasurementErrorGenerator:
    """Generator for measurement errors and state confusion."""
    
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
    
    def inject_measurement_errors(
        self,
        character_data: Dict[str, Dict[int, Any]],
        characters: List[Any],
        error_rate: float
    ) -> Dict[Tuple[str, int], ErrorRecord]:
        """Inject realistic measurement errors."""
        
        error_records = {}
        
        for taxon, char_dict in character_data.items():
            for char_idx, current_state in char_dict.items():
                if current_state == '?' or current_state is None:
                    continue  # Skip already missing data
                
                # Apply error probability
                if self.rng.random() < error_rate:
                    character = characters[char_idx]
                    
                    # Generate erroneous state
                    error_state = self._generate_error_state(character, current_state)
                    
                    if error_state != current_state:
                        record = ErrorRecord(
                            error_type="measurement",
                            taxon=taxon,
                            character_index=char_idx,
                            original_state=current_state,
                            modified_state=error_state,
                            pattern="measurement_error",
                            metadata={
                                'character_name': character.name,
                                'character_type': character.character_type.value if hasattr(character.character_type, 'value') else str(character.character_type)
                            }
                        )
                        
                        error_records[(taxon, char_idx)] = record
        
        return error_records
    
    def _generate_error_state(self, character: Any, current_state: str) -> str:
        """Generate a realistic error state for a character."""
        
        available_states = [s for s in character.states if s != current_state]
        if not available_states:
            return current_state
        
        # For binary characters, just flip
        if len(character.states) == 2:
            return available_states[0]
        
        # For multistate characters, consider different error models
        if hasattr(character, 'matrix_type') and character.matrix_type == 'scribal_confusion':
            # Scribal errors - adjacent states more likely
            return self._scribal_confusion_error(character, current_state, available_states)
        else:
            # General confusion - nearby states more likely
            return self._adjacent_state_error(character, current_state, available_states)
    
    def _scribal_confusion_error(self, character: Any, current_state: str, available_states: List[str]) -> str:
        """Generate scribal confusion-style errors."""
        try:
            current_idx = character.states.index(current_state)
            
            # Prefer adjacent indices (common scribal errors)
            weights = []
            for state in available_states:
                state_idx = character.states.index(state)
                distance = abs(state_idx - current_idx)
                if distance == 1:
                    weight = 0.6  # High probability for adjacent
                elif distance == 2:
                    weight = 0.3  # Medium for nearby
                else:
                    weight = 0.1  # Low for distant
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                return self.rng.choice(available_states, p=weights)
            
        except (ValueError, IndexError):
            pass
        
        # Fallback to random
        return self.rng.choice(available_states)
    
    def _adjacent_state_error(self, character: Any, current_state: str, available_states: List[str]) -> str:
        """Generate adjacent state errors."""
        try:
            current_idx = character.states.index(current_state)
            
            # Find adjacent states
            adjacent_states = []
            for state in available_states:
                state_idx = character.states.index(state)
                if abs(state_idx - current_idx) <= 1:
                    adjacent_states.append(state)
            
            if adjacent_states:
                return self.rng.choice(adjacent_states)
            
        except (ValueError, IndexError):
            pass
        
        # Fallback to random
        return self.rng.choice(available_states)


class PolymorphismGenerator:
    """Generator for polymorphisms and ambiguous states."""
    
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
    
    def inject_polymorphisms(
        self,
        character_data: Dict[str, Dict[int, Any]],
        characters: List[Any],
        polymorphism_rate: float,
        complexity: float = 0.6
    ) -> Dict[Tuple[str, int], ErrorRecord]:
        """Inject polymorphisms (multiple states for single observation)."""
        
        polymorphism_records = {}
        
        for taxon, char_dict in character_data.items():
            for char_idx, current_state in char_dict.items():
                if current_state == '?' or current_state is None:
                    continue  # Skip missing data
                
                # Apply polymorphism probability
                if self.rng.random() < polymorphism_rate:
                    character = characters[char_idx]
                    
                    # Generate polymorphic state
                    poly_state = self._generate_polymorphic_state(
                        character, current_state, complexity
                    )
                    
                    if poly_state != current_state:
                        record = ErrorRecord(
                            error_type="polymorphism",
                            taxon=taxon,
                            character_index=char_idx,
                            original_state=current_state,
                            modified_state=poly_state,
                            pattern="polymorphism",
                            metadata={
                                'character_name': character.name,
                                'complexity': complexity
                            }
                        )
                        
                        polymorphism_records[(taxon, char_idx)] = record
        
        return polymorphism_records
    
    def _generate_polymorphic_state(self, character: Any, current_state: str, complexity: float) -> str:
        """Generate a polymorphic state representation."""
        
        available_states = [s for s in character.states if s != current_state]
        if not available_states:
            return current_state
        
        # Determine number of additional states based on complexity
        if complexity > 0.8:
            n_additional = min(2, len(available_states))
        elif complexity > 0.5:
            n_additional = 1
        else:
            if self.rng.random() < 0.7:
                n_additional = 1
            else:
                n_additional = min(2, len(available_states))
        
        # Select additional states
        additional_states = self.rng.choice(
            available_states, 
            size=min(n_additional, len(available_states)), 
            replace=False
        ).tolist()
        
        # Create polymorphic representation
        all_states = [current_state] + additional_states
        all_states.sort()  # Consistent ordering
        
        # Use different separators for different formats
        if len(all_states) == 2:
            return f"{all_states[0]}/{all_states[1]}"  # Standard ambiguity
        else:
            return f"({','.join(all_states)})"  # Multiple state polymorphism


class ErrorEngine:
    """
    Comprehensive error injection engine for realistic phylogenetic data.
    
    Implements multiple types of errors commonly found in phylogenetic datasets:
    - Missing data with various patterns (random, systematic, clustered, taxon-specific)
    - Measurement errors and observational noise
    - Polymorphisms and ambiguous character states
    - Domain-specific error patterns
    
    Maintains complete records of all errors for ground truth validation.
    """
    
    def __init__(self, config: Any, seed: int = None):
        """Initialize error injection engine."""
        self.config = config
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Initialize component generators
        self.missing_generator = MissingDataGenerator(self.rng)
        self.measurement_generator = MeasurementErrorGenerator(self.rng)
        self.polymorphism_generator = PolymorphismGenerator(self.rng)
        
        logger.debug(f"Error engine initialized with missing_rate={config.missing_data_rate}, "
                    f"measurement_rate={config.measurement_error_rate}, "
                    f"polymorphism_rate={config.polymorphism_rate}")
    
    def inject_errors(
        self, 
        character_data: Dict[str, Dict[int, Any]], 
        characters: List[Any], 
        taxa: List[str]
    ) -> Tuple[Dict[str, Dict[int, Any]], ErrorPatterns]:
        """
        Inject realistic errors into clean phylogenetic data.
        
        Args:
            character_data: Clean character data matrix
            characters: List of Character objects
            taxa: List of taxon names
            
        Returns:
            Tuple of (degraded_data, error_patterns)
        """
        logger.info(f"Injecting errors into dataset: {len(taxa)} taxa, {len(characters)} characters")
        
        # Create deep copy to avoid modifying original
        degraded_data = deepcopy(character_data)
        error_patterns = ErrorPatterns()
        
        # Step 1: Inject missing data
        if self.config.missing_data_rate > 0:
            self._inject_missing_data(degraded_data, characters, taxa, error_patterns)
        
        # Step 2: Inject measurement errors
        if self.config.measurement_error_rate > 0:
            self._inject_measurement_errors(degraded_data, characters, taxa, error_patterns)
        
        # Step 3: Inject polymorphisms
        if self.config.polymorphism_rate > 0:
            self._inject_polymorphisms(degraded_data, characters, taxa, error_patterns)
        
        # Calculate final statistics
        self._calculate_error_statistics(degraded_data, characters, taxa, error_patterns)
        
        logger.info(f"Error injection completed: {error_patterns.total_errors} total errors, "
                   f"{error_patterns.missing_rate_actual:.1%} missing data, "
                   f"{error_patterns.error_rate_actual:.1%} measurement errors, "
                   f"{error_patterns.polymorphism_rate_actual:.1%} polymorphisms")
        
        return degraded_data, error_patterns
    
    def _inject_missing_data(
        self, 
        character_data: Dict[str, Dict[int, Any]], 
        characters: List[Any], 
        taxa: List[str],
        error_patterns: ErrorPatterns
    ) -> None:
        """Inject missing data according to configured patterns."""
        
        missing_cells = set()
        
        # Apply different missing patterns
        for pattern in self.config.missing_patterns:
            pattern_rate = self.config.missing_data_rate / len(self.config.missing_patterns)
            
            if pattern == MissingPattern.RANDOM:
                pattern_missing = self.missing_generator.generate_random_missing(
                    character_data, taxa, characters, pattern_rate
                )
                
            elif pattern == MissingPattern.SYSTEMATIC:
                pattern_missing = self.missing_generator.generate_systematic_missing(
                    character_data, taxa, characters, pattern_rate, 
                    self.config.systematic_missing_bias
                )
                
            elif pattern == MissingPattern.CLUSTERED:
                pattern_missing = self.missing_generator.generate_clustered_missing(
                    character_data, taxa, characters, pattern_rate,
                    self.config.cluster_size_range
                )
                
            elif pattern == MissingPattern.TAXON_SPECIFIC:
                pattern_missing = self.missing_generator.generate_taxon_specific_missing(
                    character_data, taxa, characters, pattern_rate
                )
            
            else:
                continue
            
            missing_cells.update(pattern_missing)
        
        # Apply missing data
        for taxon, char_idx in missing_cells:
            if char_idx < len(characters):
                original_state = character_data[taxon].get(char_idx, '?')
                if original_state != '?':  # Don't overwrite existing missing
                    character_data[taxon][char_idx] = '?'
                    
                    record = ErrorRecord(
                        error_type="missing",
                        taxon=taxon,
                        character_index=char_idx,
                        original_state=str(original_state),
                        modified_state='?',
                        pattern="missing_data"
                    )
                    error_patterns.add_error(record)
    
    def _inject_measurement_errors(
        self, 
        character_data: Dict[str, Dict[int, Any]], 
        characters: List[Any], 
        taxa: List[str],
        error_patterns: ErrorPatterns
    ) -> None:
        """Inject measurement errors."""
        
        measurement_errors = self.measurement_generator.inject_measurement_errors(
            character_data, characters, self.config.measurement_error_rate
        )
        
        # Apply measurement errors
        for (taxon, char_idx), error_record in measurement_errors.items():
            character_data[taxon][char_idx] = error_record.modified_state
            error_patterns.add_error(error_record)
    
    def _inject_polymorphisms(
        self, 
        character_data: Dict[str, Dict[int, Any]], 
        characters: List[Any], 
        taxa: List[str],
        error_patterns: ErrorPatterns
    ) -> None:
        """Inject polymorphisms."""
        
        polymorphisms = self.polymorphism_generator.inject_polymorphisms(
            character_data, characters, self.config.polymorphism_rate,
            self.config.polymorphism_complexity
        )
        
        # Apply polymorphisms
        for (taxon, char_idx), poly_record in polymorphisms.items():
            character_data[taxon][char_idx] = poly_record.modified_state
            error_patterns.add_error(poly_record)
    
    def _calculate_error_statistics(
        self, 
        character_data: Dict[str, Dict[int, Any]], 
        characters: List[Any], 
        taxa: List[str],
        error_patterns: ErrorPatterns
    ) -> None:
        """Calculate actual error rates achieved."""
        
        total_cells = len(taxa) * len(characters)
        missing_count = 0
        error_count = len(error_patterns.measurement_errors)
        polymorphism_count = len(error_patterns.polymorphisms)
        
        # Count actual missing data
        for taxon_data in character_data.values():
            for state in taxon_data.values():
                if state == '?' or state is None:
                    missing_count += 1
        
        error_patterns.missing_rate_actual = missing_count / total_cells if total_cells > 0 else 0
        error_patterns.error_rate_actual = error_count / total_cells if total_cells > 0 else 0
        error_patterns.polymorphism_rate_actual = polymorphism_count / total_cells if total_cells > 0 else 0