"""
Phylogenetic tree generation for Titivillus.

This module provides comprehensive tree generation capabilities including
balanced, pectinate, and random topologies with realistic branch length
distributions and calibration points.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import random
from scipy import stats

from .config import TreeConfig


@dataclass
class TreeResult:
    """Result of tree generation."""
    taxa: List[str]
    newick: str
    branch_lengths: Dict[str, float]
    node_ages: Dict[str, float]
    internal_nodes: List[str]
    topology_info: Dict[str, Any]


class TreeGenerator:
    """Generate phylogenetic trees with specified properties."""
    
    def __init__(self, config: TreeConfig, seed: int = 42):
        """Initialize tree generator."""
        self.config = config
        self.rng = np.random.RandomState(seed)
        
    def generate(self) -> TreeResult:
        """Generate a phylogenetic tree according to configuration."""
        
        # Generate taxa names
        taxa = self._generate_taxa_names()
        
        # Generate topology
        if self.config.tree_shape == "balanced":
            newick, internal_nodes = self._generate_balanced_tree(taxa)
        elif self.config.tree_shape == "pectinate":
            newick, internal_nodes = self._generate_pectinate_tree(taxa)
        else:  # random
            newick, internal_nodes = self._generate_random_tree(taxa)
        
        # Generate branch lengths
        branch_lengths = self._generate_branch_lengths(taxa, internal_nodes)
        
        # Apply calibrations if specified
        if self.config.calibration_points:
            branch_lengths = self._apply_calibrations(branch_lengths, internal_nodes)
        
        # Calculate node ages
        node_ages = self._calculate_node_ages(branch_lengths, internal_nodes)
        
        # Create topology info
        topology_info = {
            'shape': self.config.tree_shape,
            'height': self.config.tree_height,
            'branch_length_distribution': self.config.branch_length_distribution,
            'calibration_count': len(self.config.calibration_points),
            'total_tree_length': sum(branch_lengths.values())
        }
        
        return TreeResult(
            taxa=taxa,
            newick=newick,
            branch_lengths=branch_lengths,
            node_ages=node_ages,
            internal_nodes=internal_nodes,
            topology_info=topology_info
        )
    
    def _generate_taxa_names(self) -> List[str]:
        """Generate realistic taxa names."""
        taxa_count = self.config.taxa_count
        
        # Generate names based on common patterns
        # This could be made more sophisticated with actual name databases
        prefixes = ['Taxon', 'Species', 'Lang', 'MS', 'Pop', 'Group']
        
        taxa = []
        for i in range(taxa_count):
            if taxa_count <= 26:
                # Use single letters for small trees
                taxa.append(chr(65 + i))  # A, B, C, ...
            elif taxa_count <= 676:
                # Use two letters for medium trees
                first = chr(65 + (i // 26))
                second = chr(65 + (i % 26))
                taxa.append(f"{first}{second}")
            else:
                # Use prefix + number for large trees
                prefix = self.rng.choice(prefixes)
                taxa.append(f"{prefix}_{i+1:03d}")
        
        return taxa
    
    def _generate_balanced_tree(self, taxa: List[str]) -> Tuple[str, List[str]]:
        """Generate a balanced binary tree."""
        n_taxa = len(taxa)
        
        if n_taxa == 1:
            return taxa[0], []
        elif n_taxa == 2:
            return f"({taxa[0]},{taxa[1]})", []
        
        # Recursively split taxa into two balanced groups
        mid = n_taxa // 2
        left_taxa = taxa[:mid]
        right_taxa = taxa[mid:]
        
        # Generate subtrees
        left_newick, left_internals = self._generate_balanced_tree(left_taxa)
        right_newick, right_internals = self._generate_balanced_tree(right_taxa)
        
        # Create internal node name
        internal_node = f"Node_{len(left_internals) + len(right_internals) + 1}"
        
        # Combine into parent tree
        newick = f"({left_newick},{right_newick}){internal_node}"
        internal_nodes = left_internals + right_internals + [internal_node]
        
        return newick, internal_nodes
    
    def _generate_pectinate_tree(self, taxa: List[str]) -> Tuple[str, List[str]]:
        """Generate a pectinate (ladder-like) tree."""
        if len(taxa) == 1:
            return taxa[0], []
        elif len(taxa) == 2:
            return f"({taxa[0]},{taxa[1]})", []
        
        # Build pectinate structure: ((((A,B),C),D),E)...
        newick = taxa[0]
        internal_nodes = []
        
        for i in range(1, len(taxa)):
            internal_node = f"Node_{i}"
            internal_nodes.append(internal_node)
            newick = f"({newick},{taxa[i]}){internal_node}"
        
        return newick, internal_nodes
    
    def _generate_random_tree(self, taxa: List[str]) -> Tuple[str, List[str]]:
        """Generate a random binary tree topology."""
        if len(taxa) == 1:
            return taxa[0], []
        elif len(taxa) == 2:
            return f"({taxa[0]},{taxa[1]})", []
        
        # Randomly select two items to join
        remaining = taxa.copy()
        internal_nodes = []
        node_counter = 1
        
        while len(remaining) > 1:
            # Randomly select two items
            indices = self.rng.choice(len(remaining), 2, replace=False)
            indices = sorted(indices, reverse=True)  # Remove from back to front
            
            item1 = remaining.pop(indices[0])
            item2 = remaining.pop(indices[1])
            
            # Create new internal node
            internal_node = f"Node_{node_counter}"
            internal_nodes.append(internal_node)
            node_counter += 1
            
            # Create new subtree
            new_subtree = f"({item1},{item2}){internal_node}"
            remaining.append(new_subtree)
        
        return remaining[0], internal_nodes
    
    def _generate_branch_lengths(self, taxa: List[str], internal_nodes: List[str]) -> Dict[str, float]:
        """Generate branch lengths according to specified distribution."""
        all_nodes = taxa + internal_nodes
        branch_lengths = {}
        
        if self.config.branch_length_distribution == "exponential":
            rate = self.config.branch_length_params.get("rate", 1.0)
            for node in all_nodes:
                branch_lengths[node] = self.rng.exponential(1.0 / rate)
                
        elif self.config.branch_length_distribution == "uniform":
            min_val = self.config.branch_length_params.get("min", 0.01)
            max_val = self.config.branch_length_params.get("max", 1.0)
            for node in all_nodes:
                branch_lengths[node] = self.rng.uniform(min_val, max_val)
                
        elif self.config.branch_length_distribution == "gamma":
            shape = self.config.branch_length_params.get("shape", 2.0)
            scale = self.config.branch_length_params.get("scale", 0.5)
            for node in all_nodes:
                branch_lengths[node] = self.rng.gamma(shape, scale)
                
        else:
            # Default to exponential
            for node in all_nodes:
                branch_lengths[node] = self.rng.exponential(0.5)
        
        # Scale to match desired tree height if specified
        if self.config.tree_height:
            max_length = max(branch_lengths.values())
            scale_factor = self.config.tree_height / max_length
            branch_lengths = {node: length * scale_factor 
                            for node, length in branch_lengths.items()}
        
        return branch_lengths
    
    def _apply_calibrations(self, branch_lengths: Dict[str, float], 
                          internal_nodes: List[str]) -> Dict[str, float]:
        """Apply age calibrations to specific nodes."""
        # This is a simplified implementation
        # In practice, this would require more sophisticated tree traversal
        # and constraint satisfaction
        
        calibrated_lengths = branch_lengths.copy()
        
        for calibration in self.config.calibration_points:
            node_name = calibration.get("node")
            target_age = calibration.get("age")
            
            if node_name in internal_nodes and target_age:
                # Simple scaling approach - would need more sophistication
                current_age = branch_lengths.get(node_name, 0)
                if current_age > 0:
                    scale_factor = target_age / current_age
                    calibrated_lengths[node_name] = target_age
                    
                    # Scale descendants proportionally (simplified)
                    for node in internal_nodes:
                        if node != node_name:  # Don't re-scale calibrated node
                            calibrated_lengths[node] *= scale_factor
        
        return calibrated_lengths
    
    def _calculate_node_ages(self, branch_lengths: Dict[str, float], 
                           internal_nodes: List[str]) -> Dict[str, float]:
        """Calculate node ages from branch lengths."""
        # This is a simplified calculation
        # In practice would require proper tree traversal
        
        node_ages = {}
        
        # Root age (oldest internal node)
        if internal_nodes:
            root_age = max(branch_lengths.get(node, 0) for node in internal_nodes)
            
            for node in internal_nodes:
                # Simplified age calculation
                node_ages[node] = branch_lengths.get(node, 0)
        
        # Tips are at age 0
        for taxon in [node for node in branch_lengths.keys() if node not in internal_nodes]:
            node_ages[taxon] = 0.0
        
        return node_ages