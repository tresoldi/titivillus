"""
Stub error engine for initial implementation.
"""

class ErrorEngine:
    def __init__(self, config, seed=None):
        self.config = config
        self.seed = seed
    
    def inject_errors(self, character_data, characters, taxa):
        """Stub method - returns unmodified data."""
        return character_data, {}