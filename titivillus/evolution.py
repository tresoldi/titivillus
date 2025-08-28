"""
Evolution simulation engine for Titivillus.

This module implements character evolution simulation along phylogenetic trees
using continuous-time Markov chains, discrete event simulation, and hybrid approaches.
Supports asymmetric evolution, domain-specific constraints, and realistic substitution patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import linalg
import random

from .config import SimulationMode

logger = logging.getLogger(__name__)


@dataclass
class EvolutionHistory:
    """Track evolution history for a single character."""
    character_name: str
    substitutions: List[Tuple[str, str, float, str]]  # (from_state, to_state, time, node)
    ancestral_states: Dict[str, str]  # node -> state
    final_states: Dict[str, str]  # taxon -> final_state


class TreeParser:
    """Simple Newick tree parser for evolution simulation."""
    
    @staticmethod
    def parse_tree_structure(newick: str) -> Dict[str, Any]:
        """Parse tree structure for evolution simulation."""
        # Simplified parser - in production would need full Newick support
        # For now, extract basic parent-child relationships
        
        # Remove outer parentheses and root node if present
        tree_str = newick.strip()
        if tree_str.endswith(')Node_1') or tree_str.endswith(')'):
            tree_str = tree_str.rsplit(')', 1)[0]
            if tree_str.startswith('('):
                tree_str = tree_str[1:]
        
        return {
            'structure': tree_str,
            'is_simple': True  # Flag for simple 3-taxon trees
        }


class EvolutionEngine:
    """
    Core evolution simulation engine.
    
    Simulates character evolution along phylogenetic trees using various models:
    - Continuous-time Markov chains (CTMC)
    - Discrete event simulation
    - Hybrid approaches combining both
    
    Supports:
    - Asymmetric substitution matrices
    - Variable substitution rates
    - Domain-specific evolutionary constraints
    - Ancestral state reconstruction
    """
    
    def __init__(self, mode: SimulationMode = SimulationMode.HYBRID, seed: int = None):
        """Initialize evolution engine."""
        self.mode = mode
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        logger.debug(f"Evolution engine initialized with mode: {mode.value}")
    
    def evolve_characters(self, characters: List[Any], tree_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate evolution of characters along phylogenetic tree.
        
        Args:
            characters: List of Character objects with evolutionary parameters
            tree_info: Dictionary containing tree structure and parameters
            
        Returns:
            Dictionary containing:
            - character_data: Dict mapping taxa to character states
            - histories: Dict containing detailed evolution histories
        """
        logger.debug(f"Evolving {len(characters)} characters along tree")
        
        # Extract tree information
        taxa = tree_info['taxa']
        newick = tree_info['newick']
        branch_lengths = tree_info['branch_lengths']
        reticulations = tree_info.get('reticulations', [])
        
        # Parse tree structure
        tree_structure = TreeParser.parse_tree_structure(newick)
        
        # Initialize results
        character_data = {taxon: {} for taxon in taxa}
        histories = {}
        
        # Evolve each character independently
        for char_idx, character in enumerate(characters):
            logger.debug(f"Evolving character: {character.name}")
            
            # Get or create substitution matrix
            sub_matrix = self._get_substitution_matrix(character)
            
            # Simulate evolution for this character
            char_states, char_history = self._simulate_character_evolution(
                character=character,
                substitution_matrix=sub_matrix,
                tree_structure=tree_structure,
                taxa=taxa,
                branch_lengths=branch_lengths,
                char_index=char_idx
            )
            
            # Store results
            for taxon, state in char_states.items():
                character_data[taxon][char_idx] = state
            
            histories[character.name] = char_history
        
        logger.debug("Character evolution completed")
        
        return {
            'character_data': character_data,
            'histories': histories
        }
    
    def _get_substitution_matrix(self, character: Any) -> np.ndarray:
        """Get or create substitution matrix for character."""
        if character.step_matrix is not None:
            # Use provided step matrix
            matrix = character.step_matrix.copy()
        else:
            # Create default matrix based on character properties
            n_states = len(character.states)
            matrix = self._create_default_matrix(n_states, character)
        
        # Ensure diagonal is zero (no self-transitions in rate matrix)
        np.fill_diagonal(matrix, 0)
        
        # Scale by substitution rate
        matrix *= character.substitution_rate
        
        # Set diagonal to make rows sum to zero (Q-matrix property)
        np.fill_diagonal(matrix, -np.sum(matrix, axis=1))
        
        return matrix
    
    def _create_default_matrix(self, n_states: int, character: Any) -> np.ndarray:
        """Create default substitution matrix."""
        # Base symmetric matrix
        matrix = np.ones((n_states, n_states)) * 0.1
        np.fill_diagonal(matrix, 0)
        
        # Apply asymmetric bias if specified
        if hasattr(character, 'matrix_type') and character.matrix_type == 'asymmetric':
            # Apply directional bias (higher rates for forward transitions)
            for i in range(n_states):
                for j in range(n_states):
                    if i != j:
                        if j > i:  # Forward transition
                            matrix[i, j] *= 2.0
                        else:  # Reverse transition
                            matrix[i, j] *= 0.5
        
        return matrix
    
    def _simulate_character_evolution(
        self, 
        character: Any,
        substitution_matrix: np.ndarray,
        tree_structure: Dict[str, Any],
        taxa: List[str],
        branch_lengths: Dict[str, float],
        char_index: int
    ) -> Tuple[Dict[str, str], EvolutionHistory]:
        """Simulate evolution of a single character."""
        
        n_states = len(character.states)
        
        # Choose root state (equiprobable for now)
        root_state_idx = self.rng.randint(0, n_states)
        root_state = character.states[root_state_idx]
        
        # Initialize evolution history
        history = EvolutionHistory(
            character_name=character.name,
            substitutions=[],
            ancestral_states={'root': root_state},
            final_states={}
        )
        
        # For simple trees, use direct simulation
        if tree_structure['is_simple'] and len(taxa) <= 3:
            final_states = self._simulate_simple_tree(
                character, substitution_matrix, taxa, branch_lengths, 
                root_state_idx, history
            )
        else:
            # For complex trees, use more sophisticated simulation
            final_states = self._simulate_complex_tree(
                character, substitution_matrix, taxa, branch_lengths,
                root_state_idx, history
            )
        
        return final_states, history
    
    def _simulate_simple_tree(
        self,
        character: Any,
        substitution_matrix: np.ndarray,
        taxa: List[str],
        branch_lengths: Dict[str, float],
        root_state_idx: int,
        history: EvolutionHistory
    ) -> Dict[str, str]:
        """Simulate evolution on simple trees (≤3 taxa)."""
        
        final_states = {}
        n_states = len(character.states)
        
        # For each taxon, simulate evolution from root
        for taxon in taxa:
            branch_length = branch_lengths.get(taxon, 0.1)
            
            # Use continuous-time Markov chain
            if self.mode == SimulationMode.CTMC or self.mode == SimulationMode.HYBRID:
                final_state_idx = self._ctmc_evolve(
                    substitution_matrix, root_state_idx, branch_length
                )
            else:
                # Discrete simulation
                final_state_idx = self._discrete_evolve(
                    substitution_matrix, root_state_idx, branch_length
                )
            
            final_state = character.states[final_state_idx]
            final_states[taxon] = final_state
            history.final_states[taxon] = final_state
            
            # Record substitution if state changed
            if final_state_idx != root_state_idx:
                history.substitutions.append((
                    character.states[root_state_idx],
                    final_state,
                    branch_length,
                    taxon
                ))
        
        return final_states
    
    def _simulate_complex_tree(
        self,
        character: Any,
        substitution_matrix: np.ndarray,
        taxa: List[str],
        branch_lengths: Dict[str, float],
        root_state_idx: int,
        history: EvolutionHistory
    ) -> Dict[str, str]:
        """Simulate evolution on complex trees (>3 taxa)."""
        # Simplified implementation - treats each taxon independently from root
        # In production, would implement proper tree traversal
        
        return self._simulate_simple_tree(
            character, substitution_matrix, taxa, branch_lengths, 
            root_state_idx, history
        )
    
    def _ctmc_evolve(
        self, 
        rate_matrix: np.ndarray, 
        initial_state: int, 
        time: float
    ) -> int:
        """Evolve state using continuous-time Markov chain."""
        
        try:
            # Calculate transition probability matrix: P(t) = exp(Q*t)
            prob_matrix = linalg.expm(rate_matrix * time)
            
            # Sample final state from transition probabilities
            transition_probs = prob_matrix[initial_state, :]
            
            # Ensure probabilities are valid (non-negative, sum to 1)
            transition_probs = np.maximum(transition_probs, 0)
            prob_sum = np.sum(transition_probs)
            
            if prob_sum > 0:
                transition_probs /= prob_sum
                final_state = self.rng.choice(len(transition_probs), p=transition_probs)
            else:
                # Fallback: no change
                final_state = initial_state
                
        except (linalg.LinAlgError, ValueError):
            # Fallback for numerical issues
            logger.warning("CTMC simulation failed, using discrete approximation")
            final_state = self._discrete_evolve(rate_matrix, initial_state, time)
        
        return final_state
    
    def _discrete_evolve(
        self, 
        rate_matrix: np.ndarray, 
        initial_state: int, 
        time: float
    ) -> int:
        """Evolve state using discrete approximation."""
        
        current_state = initial_state
        n_steps = max(1, int(time * 10))  # 10 steps per time unit
        step_size = time / n_steps
        
        for _ in range(n_steps):
            # Calculate transition probabilities for this step
            rates = rate_matrix[current_state, :].copy()
            rates[current_state] = 0  # Remove self-transition
            
            # Convert rates to probabilities
            total_rate = np.sum(np.maximum(rates, 0))
            
            if total_rate > 0:
                # Probability of any transition in this step
                transition_prob = min(1.0, total_rate * step_size)
                
                if self.rng.random() < transition_prob:
                    # A transition occurs - choose which state
                    transition_probs = np.maximum(rates, 0) / total_rate
                    current_state = self.rng.choice(len(rates), p=transition_probs)
        
        return current_state
    
    def get_ancestral_reconstruction(self, character_data: Dict[str, Dict[int, Any]], 
                                   characters: List[Any], taxa: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Perform ancestral state reconstruction.
        
        Args:
            character_data: Observed character states
            characters: Character definitions  
            taxa: List of taxa
            
        Returns:
            Dictionary of reconstructed ancestral states
        """
        reconstructed = {}
        
        for char_idx, character in enumerate(characters):
            # Simple reconstruction: most common state at root
            states_count = {}
            
            for taxon in taxa:
                if taxon in character_data and char_idx in character_data[taxon]:
                    state = character_data[taxon][char_idx]
                    states_count[state] = states_count.get(state, 0) + 1
            
            if states_count:
                # Most parsimonious root state
                root_state = max(states_count.keys(), key=states_count.get)
                reconstructed[character.name] = {
                    'root': root_state,
                    'confidence': max(states_count.values()) / len(taxa)
                }
        
        return reconstructed