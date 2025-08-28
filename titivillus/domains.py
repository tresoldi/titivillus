"""
Domain-specific character generators for Titivillus.

This module provides specialized generators for different phylogenetic domains:
linguistics, stemmatology, and cultural evolution. Each domain has unique
evolutionary processes and character types.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import random

from .config import Domain, DomainConfig


class CharacterType(Enum):
    """Types of phylogenetic characters."""
    BINARY = "binary"
    MULTISTATE = "multistate"
    CONTINUOUS = "continuous"
    ORDERED = "ordered"
    UNORDERED = "unordered"


@dataclass
class Character:
    """Definition of a phylogenetic character."""
    name: str
    description: str
    character_type: CharacterType
    states: List[str]
    state_labels: Optional[List[str]] = None
    
    # Evolution parameters
    step_matrix: Optional[np.ndarray] = None
    matrix_type: str = "symmetric"  # symmetric, asymmetric
    substitution_rate: float = 1.0
    
    # Domain-specific properties
    domain: str = "unknown"
    borrowability: float = 0.5
    stability: float = 0.5
    bias_potential: str = "none"  # none, weak, strong
    
    # Missing data propensity
    missing_propensity: float = 0.0
    
    # Additional metadata
    partition: str = "default"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DomainGenerator(ABC):
    """Abstract base class for domain-specific character generators."""
    
    def __init__(self, config: DomainConfig, seed: int = 42):
        """Initialize domain generator."""
        self.config = config
        self.rng = np.random.RandomState(seed)
        
    @abstractmethod
    def generate_characters(self, count: int) -> List[Character]:
        """Generate domain-specific characters."""
        pass
    
    def _create_step_matrix(self, n_states: int, asymmetry_factor: float = 1.0) -> np.ndarray:
        """Create step matrix for character evolution."""
        matrix = np.ones((n_states, n_states))
        np.fill_diagonal(matrix, 0)
        
        if asymmetry_factor != 1.0:
            # Apply asymmetric bias
            for i in range(n_states):
                for j in range(n_states):
                    if i != j:
                        if i < j:  # Forward transitions
                            matrix[i, j] *= asymmetry_factor
                        else:  # Reverse transitions
                            matrix[i, j] *= (1.0 / asymmetry_factor)
        
        return matrix


class LinguisticDomain(DomainGenerator):
    """Generator for linguistic phylogenetic characters."""
    
    def generate_characters(self, count: int) -> List[Character]:
        """Generate linguistic characters with realistic properties."""
        characters = []
        
        # Character type distribution for linguistics
        type_weights = [0.4, 0.35, 0.2, 0.05]  # binary, multistate_small, multistate_large, ordered
        
        for i in range(count):
            char_type = self.rng.choice(
                ["binary", "multistate_small", "multistate_large", "ordered"],
                p=type_weights
            )
            
            if char_type == "binary":
                char = self._create_binary_linguistic_character(i)
            elif char_type == "multistate_small":
                char = self._create_small_multistate_character(i)
            elif char_type == "multistate_large":
                char = self._create_large_multistate_character(i)
            else:  # ordered
                char = self._create_ordered_character(i)
            
            # Apply linguistic-specific properties
            self._apply_linguistic_properties(char)
            characters.append(char)
        
        return characters
    
    def _create_binary_linguistic_character(self, index: int) -> Character:
        """Create binary linguistic character (presence/absence)."""
        templates = [
            {
                "name": f"voicing_{index}",
                "description": "Presence of voicing contrast",
                "states": ["0", "1"],
                "state_labels": ["absent", "present"],
                "borrowability": 0.3,
                "stability": 0.7
            },
            {
                "name": f"nasalization_{index}",
                "description": "Nasal vowels present",
                "states": ["0", "1"], 
                "state_labels": ["absent", "present"],
                "borrowability": 0.4,
                "stability": 0.6
            },
            {
                "name": f"case_marking_{index}",
                "description": "Case marking on nouns",
                "states": ["0", "1"],
                "state_labels": ["absent", "present"],
                "borrowability": 0.2,
                "stability": 0.8
            }
        ]
        
        template = self.rng.choice(templates)
        
        return Character(
            name=template["name"],
            description=template["description"],
            character_type=CharacterType.BINARY,
            states=template["states"],
            state_labels=template["state_labels"],
            domain="linguistics",
            borrowability=template["borrowability"],
            stability=template["stability"],
            partition="linguistics"
        )
    
    def _create_small_multistate_character(self, index: int) -> Character:
        """Create small multistate character (3-4 states)."""
        templates = [
            {
                "name": f"word_order_{index}",
                "description": "Basic word order",
                "states": ["0", "1", "2"],
                "state_labels": ["SOV", "SVO", "VSO"],
                "borrowability": 0.3,
                "stability": 0.8
            },
            {
                "name": f"syllable_structure_{index}",
                "description": "Syllable complexity",
                "states": ["0", "1", "2", "3"],
                "state_labels": ["CV", "CVC", "CCV", "CCVC"],
                "borrowability": 0.5,
                "stability": 0.6
            }
        ]
        
        template = self.rng.choice(templates)
        
        return Character(
            name=template["name"],
            description=template["description"],
            character_type=CharacterType.MULTISTATE,
            states=template["states"],
            state_labels=template["state_labels"],
            domain="linguistics",
            borrowability=template["borrowability"],
            stability=template["stability"],
            partition="linguistics"
        )
    
    def _create_large_multistate_character(self, index: int) -> Character:
        """Create large multistate character (5+ states)."""
        n_states = self.rng.randint(5, 9)
        states = [str(i) for i in range(n_states)]
        
        return Character(
            name=f"phoneme_inventory_{index}",
            description=f"Phoneme class with {n_states} variants",
            character_type=CharacterType.MULTISTATE,
            states=states,
            state_labels=[f"variant_{i}" for i in range(n_states)],
            domain="linguistics",
            borrowability=self.rng.uniform(0.3, 0.7),
            stability=self.rng.uniform(0.4, 0.8),
            partition="linguistics"
        )
    
    def _create_ordered_character(self, index: int) -> Character:
        """Create ordered multistate character."""
        templates = [
            {
                "name": f"tone_levels_{index}",
                "description": "Number of tone levels",
                "states": ["0", "1", "2", "3"],
                "state_labels": ["no_tone", "two_tone", "three_tone", "four_plus"],
                "borrowability": 0.4,
                "stability": 0.7
            }
        ]
        
        template = self.rng.choice(templates)
        
        return Character(
            name=template["name"],
            description=template["description"],
            character_type=CharacterType.ORDERED,
            states=template["states"],
            state_labels=template["state_labels"],
            domain="linguistics",
            borrowability=template["borrowability"],
            stability=template["stability"],
            partition="linguistics"
        )
    
    def _apply_linguistic_properties(self, char: Character) -> None:
        """Apply linguistic-specific properties to character."""
        # Asymmetric evolution based on configuration
        if self.config.asymmetric_bias > 0:
            asymmetry = 1.0 + (self.config.asymmetric_bias * 2.0)  # 1.0 to 3.0
            char.step_matrix = self._create_step_matrix(len(char.states), asymmetry)
            char.matrix_type = "asymmetric"
        
        # Apply borrowability from configuration
        borrowability_range = self.config.domain_params.get('borrowability_range', [0.1, 0.8])
        char.borrowability = self.rng.uniform(*borrowability_range)
        
        # Phonological constraints affect substitution rates
        if self.config.domain_params.get('phonological_constraints', False):
            char.substitution_rate *= self.rng.uniform(0.5, 1.5)


class StemmatologyDomain(DomainGenerator):
    """Generator for stemmatological characters (manuscript variants)."""
    
    def generate_characters(self, count: int) -> List[Character]:
        """Generate stemmatological characters representing textual variants."""
        characters = []
        
        for i in range(count):
            # Most stemmatological characters are multistate (different readings)
            if self.rng.random() < 0.7:
                char = self._create_textual_variant(i)
            else:
                char = self._create_binary_variant(i)
            
            self._apply_stemmatology_properties(char)
            characters.append(char)
        
        return characters
    
    def _create_textual_variant(self, index: int) -> Character:
        """Create multistate textual variant character."""
        n_variants = self.rng.randint(2, 6)  # 2-5 different readings
        states = [str(i) for i in range(n_variants)]
        
        variant_types = [
            "lexical_variant", "orthographic_variant", "grammatical_variant",
            "scribal_error", "correction", "omission_addition"
        ]
        
        variant_type = self.rng.choice(variant_types)
        
        return Character(
            name=f"{variant_type}_{index}",
            description=f"Textual variant: {variant_type}",
            character_type=CharacterType.MULTISTATE,
            states=states,
            state_labels=[f"reading_{i}" for i in range(n_variants)],
            domain="stemmatology",
            partition="stemmatology",
            metadata={"variant_type": variant_type}
        )
    
    def _create_binary_variant(self, index: int) -> Character:
        """Create binary presence/absence variant."""
        binary_types = [
            "text_present", "gloss_present", "marginal_note", 
            "illumination", "rubrication", "punctuation"
        ]
        
        variant_type = self.rng.choice(binary_types)
        
        return Character(
            name=f"{variant_type}_{index}",
            description=f"Presence of {variant_type}",
            character_type=CharacterType.BINARY,
            states=["0", "1"],
            state_labels=["absent", "present"],
            domain="stemmatology",
            partition="stemmatology",
            metadata={"variant_type": variant_type}
        )
    
    def _apply_stemmatology_properties(self, char: Character) -> None:
        """Apply stemmatology-specific properties."""
        # High error rates for error-prone positions
        error_prone_rate = self.config.domain_params.get('error_prone_positions', 0.2)
        if self.rng.random() < error_prone_rate:
            char.missing_propensity = 0.3
            char.substitution_rate *= 2.0
            char.metadata['error_prone'] = True
        
        # Scribal error patterns
        if self.config.domain_params.get('scribal_error_patterns', False):
            char.step_matrix = self._create_scribal_confusion_matrix(len(char.states))
            char.matrix_type = "scribal_confusion"
        
        # Manuscript quality affects character reliability
        quality_variation = self.config.domain_params.get('manuscript_quality_variation', 0.4)
        char.weight = self.rng.uniform(1.0 - quality_variation, 1.0 + quality_variation)
    
    def _create_scribal_confusion_matrix(self, n_states: int) -> np.ndarray:
        """Create confusion matrix based on scribal error patterns."""
        matrix = np.ones((n_states, n_states)) * 0.1  # Low base rate
        np.fill_diagonal(matrix, 0)
        
        # Higher rates for similar readings (simplified model)
        for i in range(n_states):
            for j in range(n_states):
                if abs(i - j) == 1:  # Adjacent variants more likely confused
                    matrix[i, j] = 0.3
        
        return matrix


class CulturalDomain(DomainGenerator):
    """Generator for cultural evolution characters."""
    
    def generate_characters(self, count: int) -> List[Character]:
        """Generate cultural trait characters."""
        characters = []
        
        for i in range(count):
            trait_type = self.rng.choice([
                "material_culture", "social_organization", "ritual_practice",
                "technological_trait", "artistic_style", "subsistence_strategy"
            ])
            
            char = self._create_cultural_trait(i, trait_type)
            self._apply_cultural_properties(char)
            characters.append(char)
        
        return characters
    
    def _create_cultural_trait(self, index: int, trait_type: str) -> Character:
        """Create cultural trait character."""
        if trait_type in ["material_culture", "technological_trait"]:
            # Often binary (present/absent) or few states
            if self.rng.random() < 0.6:
                states = ["0", "1"]
                state_labels = ["absent", "present"]
                char_type = CharacterType.BINARY
            else:
                n_states = self.rng.randint(3, 5)
                states = [str(i) for i in range(n_states)]
                state_labels = [f"variant_{i}" for i in range(n_states)]
                char_type = CharacterType.MULTISTATE
        else:
            # Social/ritual traits often multistate
            n_states = self.rng.randint(3, 6)
            states = [str(i) for i in range(n_states)]
            state_labels = [f"form_{i}" for i in range(n_states)]
            char_type = CharacterType.MULTISTATE
        
        return Character(
            name=f"{trait_type}_{index}",
            description=f"Cultural trait: {trait_type}",
            character_type=char_type,
            states=states,
            state_labels=state_labels,
            domain="cultural",
            partition="cultural",
            metadata={"trait_type": trait_type}
        )
    
    def _apply_cultural_properties(self, char: Character) -> None:
        """Apply cultural evolution-specific properties."""
        # Innovation bias affects directionality
        innovation_bias = self.config.domain_params.get('innovation_bias', 0.4)
        if innovation_bias > 0.5:
            # Bias toward higher state numbers (innovation)
            asymmetry = 1.0 + innovation_bias
            char.step_matrix = self._create_step_matrix(len(char.states), asymmetry)
            char.matrix_type = "innovation_biased"
        
        # Prestige effects
        prestige = self.config.domain_params.get('prestige_effects', 0.3)
        if self.rng.random() < prestige:
            char.borrowability = 0.8  # High borrowability for prestige traits
            char.metadata['prestige_trait'] = True
        
        # Horizontal transmission rate
        ht_rate = self.config.domain_params.get('horizontal_transmission', 0.5)
        char.borrowability = min(char.borrowability + ht_rate * 0.5, 1.0)


class DomainFactory:
    """Factory for creating domain-specific generators."""
    
    def __init__(self, seed: int = 42):
        """Initialize factory."""
        self.seed = seed
    
    def get_generator(self, domain: Domain, config: DomainConfig) -> DomainGenerator:
        """Get appropriate generator for domain."""
        if domain == Domain.LINGUISTICS:
            return LinguisticDomain(config, self.seed)
        elif domain == Domain.STEMMATOLOGY:
            return StemmatologyDomain(config, self.seed)
        elif domain == Domain.CULTURAL:
            return CulturalDomain(config, self.seed)
        else:
            raise ValueError(f"Unknown domain: {domain}")
    
    def list_domains(self) -> List[Domain]:
        """List available domains."""
        return list(Domain)