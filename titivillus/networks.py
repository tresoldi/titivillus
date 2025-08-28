"""
Network effects engine for Titivillus synthetic phylogenetic data.

This module implements complex network effects in phylogenetic evolution:
- Reticulation events (hybridization, horizontal gene transfer)
- Contact zones and geographic proximity effects
- Borrowing networks (linguistic borrowing, cultural diffusion, manuscript contamination)
- Domain-specific network patterns reflecting real-world evolutionary processes
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from copy import deepcopy
import random
import re

logger = logging.getLogger(__name__)


@dataclass
class ReticulationEvent:
    """Record of a reticulation event in the phylogenetic network."""
    source_taxon: str
    target_taxon: str
    strength: float  # 0.0-1.0, strength of reticulation
    time_point: float  # When in evolutionary time the event occurred
    affected_characters: List[int] = field(default_factory=list)
    event_type: str = "hybridization"  # hybridization, horizontal_transfer, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class ContactZone:
    """Record of a contact zone affecting multiple taxa."""
    taxa: List[str]
    strength: float  # 0.0-1.0, strength of contact effects
    duration: float  # Length of contact period
    geographic_center: Optional[Tuple[float, float]] = None  # lat, lng if applicable
    affected_characters: List[int] = field(default_factory=list)
    contact_type: str = "geographic"  # geographic, cultural, trade_route, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BorrowingEvent:
    """Record of a specific borrowing event between taxa."""
    donor_taxon: str
    recipient_taxon: str
    character_index: int
    original_state: str
    borrowed_state: str
    borrowing_strength: float
    time_point: float
    borrowing_type: str = "lateral_transfer"
    metadata: Dict[str, Any] = field(default_factory=dict)


class NetworkPatterns:
    """Container for all network effects applied to a dataset."""
    
    def __init__(self):
        self.reticulations: List[ReticulationEvent] = []
        self.contact_zones: List[ContactZone] = []
        self.borrowing_events: List[BorrowingEvent] = []
        
        # Summary statistics
        self.total_network_events: int = 0
        self.reticulation_rate_actual: float = 0.0
        self.contact_coverage: float = 0.0  # Proportion of taxa affected by contact
        self.borrowing_rate_actual: float = 0.0


class TreeModifier:
    """Modifies phylogenetic trees to incorporate network effects."""
    
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
    
    def add_reticulation_to_tree(
        self, 
        newick: str, 
        reticulation: ReticulationEvent
    ) -> str:
        """Add reticulation notation to Newick tree string."""
        
        # For now, return original tree with comment notation
        # In production, would implement full network Newick format
        
        # Add reticulation as comment in extended Newick
        if '[' not in newick:
            # Simple case: add reticulation info as comment
            source_pattern = f"({reticulation.source_taxon}"
            target_pattern = f"({reticulation.target_taxon}"
            
            # Add reticulation strength as comment
            comment = f"[&reticulation={reticulation.source_taxon}->{reticulation.target_taxon}:{reticulation.strength:.2f}]"
            
            # Insert comment near root for now
            if newick.endswith(';'):
                modified = newick[:-1] + comment + ';'
            else:
                modified = newick + comment
            
            return modified
        
        return newick  # Return original if already has annotations
    
    def calculate_taxa_distances(self, taxa: List[str]) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise distances between taxa for contact zone modeling."""
        
        # Simple distance model - in production would use actual branch lengths
        distances = {}
        
        for i, taxon1 in enumerate(taxa):
            for j, taxon2 in enumerate(taxa):
                if i != j:
                    # Simple alphabetical distance as proxy
                    distance = abs(ord(taxon1[0]) - ord(taxon2[0])) / 26.0
                    distances[(taxon1, taxon2)] = distance
        
        return distances


class ReticulationGenerator:
    """Generator for reticulation events (hybridization, horizontal transfer)."""
    
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
    
    def generate_reticulations(
        self, 
        taxa: List[str],
        tree_info: Dict[str, Any],
        reticulation_rate: float,
        max_reticulations: int,
        reticulation_strength: float
    ) -> List[ReticulationEvent]:
        """Generate reticulation events for the phylogeny."""
        
        reticulations = []
        
        # Calculate number of reticulation events
        n_possible = len(taxa) * (len(taxa) - 1) // 2  # Pairs of taxa
        n_reticulations = min(
            int(n_possible * reticulation_rate),
            max_reticulations
        )
        
        if n_reticulations == 0:
            return reticulations
        
        # Select taxa pairs for reticulation
        taxa_pairs = []
        for i in range(len(taxa)):
            for j in range(i + 1, len(taxa)):
                taxa_pairs.append((taxa[i], taxa[j]))
        
        selected_pairs = self.rng.choice(
            len(taxa_pairs), 
            size=min(n_reticulations, len(taxa_pairs)), 
            replace=False
        )
        
        for pair_idx in selected_pairs:
            taxon1, taxon2 = taxa_pairs[pair_idx]
            
            # Randomly decide direction of reticulation
            if self.rng.random() < 0.5:
                source, target = taxon1, taxon2
            else:
                source, target = taxon2, taxon1
            
            # Generate reticulation parameters
            strength = self.rng.uniform(0.1, reticulation_strength)
            time_point = self.rng.uniform(0.1, 0.9)  # Relative time along branches
            
            event = ReticulationEvent(
                source_taxon=source,
                target_taxon=target,
                strength=strength,
                time_point=time_point,
                event_type="hybridization",
                metadata={'pair_index': pair_idx}
            )
            
            reticulations.append(event)
        
        logger.debug(f"Generated {len(reticulations)} reticulation events")
        return reticulations


class ContactZoneGenerator:
    """Generator for contact zones affecting multiple taxa."""
    
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
    
    def generate_contact_zones(
        self, 
        taxa: List[str],
        n_contact_zones: int,
        contact_strength: float,
        duration_range: Tuple[float, float] = (0.1, 0.5)
    ) -> List[ContactZone]:
        """Generate contact zones affecting groups of taxa."""
        
        contact_zones = []
        
        if n_contact_zones == 0 or len(taxa) < 2:
            return contact_zones
        
        for _ in range(n_contact_zones):
            # Select taxa for this contact zone
            zone_size = self.rng.randint(2, max(3, len(taxa) // 2 + 1))
            zone_taxa = self.rng.choice(
                taxa, 
                size=min(zone_size, len(taxa)), 
                replace=False
            ).tolist()
            
            # Generate contact parameters
            strength = self.rng.uniform(0.1, contact_strength)
            duration = self.rng.uniform(*duration_range)
            
            # Generate contact type based on context
            contact_types = ["geographic", "cultural", "trade_route", "migration"]
            contact_type = self.rng.choice(contact_types)
            
            zone = ContactZone(
                taxa=zone_taxa,
                strength=strength,
                duration=duration,
                contact_type=contact_type,
                metadata={'zone_id': len(contact_zones)}
            )
            
            contact_zones.append(zone)
        
        logger.debug(f"Generated {len(contact_zones)} contact zones")
        return contact_zones


class BorrowingSimulator:
    """Simulates character borrowing through network connections."""
    
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
    
    def simulate_borrowing_effects(
        self,
        character_data: Dict[str, Dict[int, Any]],
        characters: List[Any],
        taxa: List[str],
        reticulations: List[ReticulationEvent],
        contact_zones: List[ContactZone],
        borrowing_rate: float
    ) -> Tuple[Dict[str, Dict[int, Any]], List[BorrowingEvent]]:
        """Simulate character borrowing through network connections."""
        
        borrowing_events = []
        modified_data = deepcopy(character_data)
        
        # Apply reticulation-based borrowing
        for reticulation in reticulations:
            borrowing_events.extend(
                self._apply_reticulation_borrowing(
                    modified_data, characters, reticulation, borrowing_rate
                )
            )
        
        # Apply contact zone borrowing
        for contact_zone in contact_zones:
            borrowing_events.extend(
                self._apply_contact_zone_borrowing(
                    modified_data, characters, contact_zone, borrowing_rate
                )
            )
        
        logger.debug(f"Simulated {len(borrowing_events)} borrowing events")
        return modified_data, borrowing_events
    
    def _apply_reticulation_borrowing(
        self,
        character_data: Dict[str, Dict[int, Any]],
        characters: List[Any],
        reticulation: ReticulationEvent,
        base_rate: float
    ) -> List[BorrowingEvent]:
        """Apply borrowing through reticulation connection."""
        
        events = []
        donor = reticulation.source_taxon
        recipient = reticulation.target_taxon
        
        # Borrowing rate influenced by reticulation strength
        effective_rate = base_rate * reticulation.strength
        
        for char_idx, character in enumerate(characters):
            if self.rng.random() < effective_rate:
                # Check if character is borrowable
                borrowability = getattr(character, 'borrowability', 0.5)
                
                if self.rng.random() < borrowability:
                    # Perform borrowing
                    donor_state = character_data[donor].get(char_idx)
                    recipient_state = character_data[recipient].get(char_idx)
                    
                    if (donor_state is not None and recipient_state is not None 
                        and donor_state != recipient_state and donor_state != '?'):
                        
                        event = BorrowingEvent(
                            donor_taxon=donor,
                            recipient_taxon=recipient,
                            character_index=char_idx,
                            original_state=str(recipient_state),
                            borrowed_state=str(donor_state),
                            borrowing_strength=reticulation.strength,
                            time_point=reticulation.time_point,
                            borrowing_type="reticulation",
                            metadata={
                                'character_name': character.name,
                                'borrowability': borrowability
                            }
                        )
                        
                        # Apply borrowing
                        character_data[recipient][char_idx] = donor_state
                        events.append(event)
        
        return events
    
    def _apply_contact_zone_borrowing(
        self,
        character_data: Dict[str, Dict[int, Any]],
        characters: List[Any],
        contact_zone: ContactZone,
        base_rate: float
    ) -> List[BorrowingEvent]:
        """Apply borrowing within contact zone."""
        
        events = []
        zone_taxa = contact_zone.taxa
        
        if len(zone_taxa) < 2:
            return events
        
        # Contact zone creates borrowing opportunities between all members
        effective_rate = base_rate * contact_zone.strength * contact_zone.duration
        
        for char_idx, character in enumerate(characters):
            if self.rng.random() < effective_rate:
                # Check borrowability
                borrowability = getattr(character, 'borrowability', 0.5)
                
                # Domain-specific borrowing patterns
                domain_modifier = self._get_domain_borrowing_modifier(
                    character, contact_zone.contact_type
                )
                adjusted_borrowability = min(borrowability * domain_modifier, 1.0)
                
                if self.rng.random() < adjusted_borrowability:
                    # Select donor and recipient from zone
                    donor = self.rng.choice(zone_taxa)
                    recipient = self.rng.choice([t for t in zone_taxa if t != donor])
                    
                    donor_state = character_data[donor].get(char_idx)
                    recipient_state = character_data[recipient].get(char_idx)
                    
                    if (donor_state is not None and recipient_state is not None 
                        and donor_state != recipient_state and donor_state != '?'):
                        
                        event = BorrowingEvent(
                            donor_taxon=donor,
                            recipient_taxon=recipient,
                            character_index=char_idx,
                            original_state=str(recipient_state),
                            borrowed_state=str(donor_state),
                            borrowing_strength=contact_zone.strength,
                            time_point=0.5,  # Contact zones occur mid-evolution
                            borrowing_type=f"contact_{contact_zone.contact_type}",
                            metadata={
                                'character_name': character.name,
                                'contact_zone_id': contact_zone.metadata.get('zone_id'),
                                'domain_modifier': domain_modifier
                            }
                        )
                        
                        # Apply borrowing
                        character_data[recipient][char_idx] = donor_state
                        events.append(event)
        
        return events
    
    def _get_domain_borrowing_modifier(self, character: Any, contact_type: str) -> float:
        """Get domain-specific borrowing rate modifier."""
        
        domain = getattr(character, 'domain', 'unknown')
        
        # Domain-specific borrowing patterns
        if domain == 'linguistics':
            if contact_type == 'geographic':
                return 1.5  # Geographic contact increases linguistic borrowing
            elif contact_type == 'trade_route':
                return 2.0  # Trade contact highly favors linguistic borrowing
            elif contact_type == 'cultural':
                return 1.3
            else:
                return 1.0
                
        elif domain == 'cultural':
            if contact_type == 'cultural':
                return 2.0  # Cultural contact favors cultural traits
            elif contact_type == 'trade_route':
                return 1.8
            elif contact_type == 'geographic':
                return 1.2
            else:
                return 1.0
                
        elif domain == 'stemmatology':
            if contact_type == 'cultural':
                return 0.3  # Manuscript contamination less likely via cultural contact
            elif contact_type == 'geographic':
                return 0.5  # Some geographic clustering of manuscripts
            else:
                return 0.2  # Generally low contamination
        
        return 1.0  # Default modifier


class NetworkEngine:
    """
    Comprehensive network effects engine for phylogenetic data simulation.
    
    Implements complex network effects that occur in real evolutionary systems:
    - Reticulation events: hybridization, horizontal gene transfer, admixture
    - Contact zones: geographic proximity, cultural contact, trade routes
    - Borrowing networks: linguistic borrowing, cultural diffusion, manuscript contamination
    - Domain-specific patterns: realistic network effects for different data types
    
    Maintains complete records of all network events for ground truth validation.
    """
    
    def __init__(self, config: Any, seed: int = None):
        """Initialize network effects engine."""
        self.config = config
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Initialize component generators
        self.tree_modifier = TreeModifier(self.rng)
        self.reticulation_generator = ReticulationGenerator(self.rng)
        self.contact_generator = ContactZoneGenerator(self.rng)
        self.borrowing_simulator = BorrowingSimulator(self.rng)
        
        logger.debug(f"Network engine initialized with reticulation_rate={config.reticulation_rate}, "
                    f"contact_zones={config.contact_zones}, borrowing_rate={config.borrowing_rate}")
    
    def apply_network_effects(
        self, 
        tree_result: Any, 
        taxa: List[str]
    ) -> Dict[str, Any]:
        """
        Apply network effects to phylogenetic tree and prepare for character borrowing.
        
        Args:
            tree_result: TreeResult object with original phylogeny
            taxa: List of taxon names
            
        Returns:
            Dictionary containing modified tree and network event records
        """
        logger.info(f"Applying network effects to {len(taxa)} taxa")
        
        network_patterns = NetworkPatterns()
        modified_newick = tree_result.newick
        
        # Step 1: Generate reticulation events
        if self.config.enable_reticulation and self.config.reticulation_rate > 0:
            reticulations = self.reticulation_generator.generate_reticulations(
                taxa=taxa,
                tree_info={'newick': tree_result.newick, 'branch_lengths': tree_result.branch_lengths},
                reticulation_rate=self.config.reticulation_rate,
                max_reticulations=self.config.max_reticulations,
                reticulation_strength=self.config.reticulation_strength
            )
            network_patterns.reticulations = reticulations
            
            # Modify tree notation to include reticulations
            for reticulation in reticulations:
                modified_newick = self.tree_modifier.add_reticulation_to_tree(
                    modified_newick, reticulation
                )
        
        # Step 2: Generate contact zones
        if self.config.contact_zones > 0:
            contact_zones = self.contact_generator.generate_contact_zones(
                taxa=taxa,
                n_contact_zones=self.config.contact_zones,
                contact_strength=self.config.contact_strength,
                duration_range=self.config.contact_duration_range
            )
            network_patterns.contact_zones = contact_zones
        
        # Calculate summary statistics
        network_patterns.total_network_events = (
            len(network_patterns.reticulations) + len(network_patterns.contact_zones)
        )
        network_patterns.reticulation_rate_actual = len(network_patterns.reticulations) / max(len(taxa) - 1, 1)
        
        # Calculate contact coverage
        contacted_taxa = set()
        for zone in network_patterns.contact_zones:
            contacted_taxa.update(zone.taxa)
        network_patterns.contact_coverage = len(contacted_taxa) / len(taxa) if taxa else 0.0
        
        logger.info(f"Network effects applied: {len(network_patterns.reticulations)} reticulations, "
                   f"{len(network_patterns.contact_zones)} contact zones, "
                   f"{network_patterns.contact_coverage:.1%} taxa in contact")
        
        return {
            'modified_tree': modified_newick,
            'reticulations': network_patterns.reticulations,
            'contact_zones': network_patterns.contact_zones,
            'network_patterns': network_patterns
        }
    
    def apply_character_borrowing(
        self,
        character_data: Dict[str, Dict[int, Any]],
        characters: List[Any],
        taxa: List[str],
        reticulations: List[ReticulationEvent],
        contact_zones: List[ContactZone]
    ) -> Tuple[Dict[str, Dict[int, Any]], List[BorrowingEvent]]:
        """
        Apply character borrowing through network connections.
        
        This method is called after character evolution to simulate
        horizontal transfer of characters through network connections.
        
        Args:
            character_data: Evolved character data matrix
            characters: List of Character objects
            taxa: List of taxon names
            reticulations: Reticulation events from network effects
            contact_zones: Contact zones from network effects
            
        Returns:
            Tuple of (modified_character_data, borrowing_events)
        """
        if not reticulations and not contact_zones:
            return character_data, []
        
        logger.info(f"Applying character borrowing through {len(reticulations)} reticulations "
                   f"and {len(contact_zones)} contact zones")
        
        modified_data, borrowing_events = self.borrowing_simulator.simulate_borrowing_effects(
            character_data=character_data,
            characters=characters,
            taxa=taxa,
            reticulations=reticulations,
            contact_zones=contact_zones,
            borrowing_rate=self.config.borrowing_rate
        )
        
        logger.info(f"Character borrowing completed: {len(borrowing_events)} borrowing events")
        return modified_data, borrowing_events