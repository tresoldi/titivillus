"""
Stub validation classes for initial implementation.
"""

class GroundTruth:
    def __init__(self, true_tree=None, branch_lengths=None, character_histories=None, 
                 error_patterns=None, reticulations=None, contact_zones=None,
                 borrowing_events=None, generation_config=None, validation_level=None):
        self.true_tree = true_tree
        self.branch_lengths = branch_lengths
        self.character_histories = character_histories or {}
        self.error_patterns = error_patterns or {}
        self.reticulations = reticulations or []
        self.contact_zones = contact_zones or []
        self.borrowing_events = borrowing_events or []
        self.generation_config = generation_config
        self.validation_level = validation_level