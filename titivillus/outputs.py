"""
Output engine for Titivillus synthetic phylogenetic data export.

This module provides comprehensive export functionality for generated datasets
in multiple formats including NEXUS, CSV, JSON, and Newick. Ensures compatibility
with phylogenetic software and analysis pipelines.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .config import OutputFormat

logger = logging.getLogger(__name__)


class NexusWriter:
    """Writer for NEXUS format files compatible with phylogenetic software."""
    
    @staticmethod
    def write(dataset: Any, output_path: Path, options: Dict[str, Any]) -> None:
        """Write dataset to NEXUS format."""
        
        with open(output_path, 'w') as f:
            f.write("#NEXUS\n\n")
            
            # Data block
            f.write("BEGIN DATA;\n")
            f.write(f"  DIMENSIONS NTAX={len(dataset.taxa)} NCHAR={len(dataset.characters)};\n")
            
            # Determine format
            datatype = NexusWriter._determine_datatype(dataset.characters)
            f.write(f"  FORMAT DATATYPE={datatype} GAP=- MISSING=?;\n")
            
            # Character labels if requested
            if options.get('include_character_labels', True):
                f.write("  CHARLABELS\n")
                for i, char in enumerate(dataset.characters):
                    f.write(f"    {i+1} '{char.name}'\n")
                f.write("  ;\n")
            
            # State labels if requested and applicable
            if options.get('include_state_labels', True) and datatype == "STANDARD":
                NexusWriter._write_state_labels(f, dataset.characters)
            
            # Data matrix
            f.write("  MATRIX\n")
            for taxon in dataset.taxa:
                f.write(f"    {taxon:12s} ")
                char_states = []
                for char_idx in range(len(dataset.characters)):
                    state = dataset.character_data[taxon].get(char_idx, '?')
                    char_states.append(str(state))
                f.write("".join(char_states))
                f.write("\n")
            f.write("  ;\n")
            f.write("END;\n\n")
            
            # Trees block if requested
            if options.get('include_trees', True):
                f.write("BEGIN TREES;\n")
                f.write(f"  TREE true_tree = {dataset.tree_newick};\n")
                f.write("END;\n\n")
            
            # Assumptions block if requested
            if options.get('include_assumptions', True):
                NexusWriter._write_assumptions(f, dataset)
    
    @staticmethod
    def _determine_datatype(characters: List[Any]) -> str:
        """Determine appropriate NEXUS datatype."""
        # For now, use STANDARD for discrete characters
        return "STANDARD"
    
    @staticmethod
    def _write_state_labels(f, characters: List[Any]) -> None:
        """Write state labels for characters."""
        f.write("  STATELABELS\n")
        for i, char in enumerate(characters):
            if char.state_labels:
                f.write(f"    {i+1} {' '.join(char.state_labels)}\n")
        f.write("  ;\n")
    
    @staticmethod
    def _write_assumptions(f, dataset: Any) -> None:
        """Write assumptions block with character information."""
        f.write("BEGIN ASSUMPTIONS;\n")
        
        # Character sets by domain
        domains = {}
        for i, char in enumerate(dataset.characters):
            domain = getattr(char, 'domain', 'unknown')
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(str(i + 1))
        
        for domain, char_indices in domains.items():
            if len(char_indices) > 1:
                f.write(f"  CHARSET {domain} = {' '.join(char_indices)};\n")
        
        f.write("END;\n\n")


class CsvWriter:
    """Writer for CSV format files for statistical analysis."""
    
    @staticmethod
    def write(dataset: Any, output_path: Path, options: Dict[str, Any]) -> List[Path]:
        """Write dataset to CSV format(s)."""
        
        output_files = []
        
        # Main character data matrix
        matrix_path = output_path.with_suffix('.csv')
        CsvWriter._write_character_matrix(dataset, matrix_path)
        output_files.append(matrix_path)
        
        # Metadata file if requested
        if options.get('include_metadata', True):
            metadata_path = output_path.with_name(output_path.stem + '_metadata.csv')
            CsvWriter._write_metadata(dataset, metadata_path)
            output_files.append(metadata_path)
        
        # Separate partitions if requested
        if options.get('separate_partitions', True):
            partition_files = CsvWriter._write_partitions(dataset, output_path)
            output_files.extend(partition_files)
        
        return output_files
    
    @staticmethod
    def _write_character_matrix(dataset: Any, output_path: Path) -> None:
        """Write main character data matrix."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['taxon'] + [f'char_{i}' for i in range(len(dataset.characters))]
            writer.writerow(header)
            
            # Data rows
            for taxon in dataset.taxa:
                row = [taxon]
                for char_idx in range(len(dataset.characters)):
                    state = dataset.character_data[taxon].get(char_idx, '?')
                    row.append(state)
                writer.writerow(row)
    
    @staticmethod
    def _write_metadata(dataset: Any, output_path: Path) -> None:
        """Write character metadata."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'character_index', 'character_name', 'character_type', 
                'domain', 'states', 'substitution_rate', 'description'
            ])
            
            # Character information
            for i, char in enumerate(dataset.characters):
                writer.writerow([
                    i,
                    char.name,
                    char.character_type.value if hasattr(char.character_type, 'value') else str(char.character_type),
                    char.domain,
                    '|'.join(char.states),
                    char.substitution_rate,
                    char.description
                ])
    
    @staticmethod
    def _write_partitions(dataset: Any, base_path: Path) -> List[Path]:
        """Write separate files for each domain partition."""
        partition_files = []
        
        # Group characters by domain
        domains = {}
        for i, char in enumerate(dataset.characters):
            domain = getattr(char, 'domain', 'unknown')
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(i)
        
        # Write partition files
        for domain, char_indices in domains.items():
            if len(char_indices) > 1:  # Only create partition if multiple characters
                partition_path = base_path.with_name(f"{base_path.stem}_{domain}.csv")
                
                with open(partition_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    header = ['taxon'] + [f'{dataset.characters[i].name}' for i in char_indices]
                    writer.writerow(header)
                    
                    # Data
                    for taxon in dataset.taxa:
                        row = [taxon]
                        for char_idx in char_indices:
                            state = dataset.character_data[taxon].get(char_idx, '?')
                            row.append(state)
                        writer.writerow(row)
                
                partition_files.append(partition_path)
        
        return partition_files


class JsonWriter:
    """Writer for JSON format with complete metadata and ground truth."""
    
    @staticmethod
    def write(dataset: Any, output_path: Path, options: Dict[str, Any]) -> None:
        """Write dataset to JSON format."""
        
        # Build comprehensive JSON structure
        data = {
            'metadata': {
                'name': dataset.name,
                'generation_time': dataset.generation_time.isoformat(),
                'generator_version': '0.1.0',
                'format_version': '1.0'
            },
            'configuration': JsonWriter._serialize_config(dataset.config),
            'taxa': dataset.taxa,
            'characters': JsonWriter._serialize_characters(dataset.characters),
            'character_data': JsonWriter._serialize_character_data(dataset.character_data),
            'tree': {
                'newick': dataset.tree_newick,
                'branch_lengths': dataset.branch_lengths
            },
            'statistics': dataset.stats
        }
        
        # Add ground truth if requested
        if options.get('include_ground_truth', True):
            data['ground_truth'] = JsonWriter._serialize_ground_truth(dataset.ground_truth)
        
        # Add network information if present
        if dataset.reticulations or dataset.contact_zones:
            data['networks'] = {
                'reticulations': dataset.reticulations,
                'contact_zones': dataset.contact_zones
            }
        
        # Write JSON
        with open(output_path, 'w') as f:
            if options.get('compact', False):
                json.dump(data, f, separators=(',', ':'))
            else:
                json.dump(data, f, indent=2, sort_keys=False)
    
    @staticmethod
    def _serialize_config(config: Any) -> Dict[str, Any]:
        """Serialize configuration object."""
        return {
            'name': config.name,
            'seed': config.seed,
            'simulation_mode': config.simulation_mode.value if hasattr(config.simulation_mode, 'value') else str(config.simulation_mode),
            'active_domains': [d.value if hasattr(d, 'value') else str(d) for d in config.active_domains],
            'tree_parameters': {
                'taxa_count': config.tree.taxa_count,
                'tree_height': config.tree.tree_height,
                'tree_shape': config.tree.tree_shape
            }
        }
    
    @staticmethod
    def _serialize_characters(characters: List[Any]) -> List[Dict[str, Any]]:
        """Serialize character objects."""
        result = []
        for char in characters:
            char_data = {
                'name': char.name,
                'description': char.description,
                'character_type': char.character_type.value if hasattr(char.character_type, 'value') else str(char.character_type),
                'states': char.states,
                'state_labels': char.state_labels,
                'domain': char.domain,
                'substitution_rate': char.substitution_rate,
                'matrix_type': char.matrix_type
            }
            if hasattr(char, 'borrowability'):
                char_data['borrowability'] = char.borrowability
            if hasattr(char, 'stability'):
                char_data['stability'] = char.stability
            result.append(char_data)
        return result
    
    @staticmethod
    def _serialize_character_data(character_data: Dict[str, Dict[int, Any]]) -> Dict[str, Dict[str, Any]]:
        """Serialize character data matrix."""
        # Convert integer keys to strings for JSON compatibility
        result = {}
        for taxon, char_dict in character_data.items():
            result[taxon] = {str(k): v for k, v in char_dict.items()}
        return result
    
    @staticmethod
    def _serialize_ground_truth(ground_truth: Any) -> Dict[str, Any]:
        """Serialize ground truth information."""
        result = {
            'true_tree': ground_truth.true_tree,
            'branch_lengths': ground_truth.branch_lengths
        }
        
        # Serialize evolution histories if present
        if hasattr(ground_truth, 'character_histories') and ground_truth.character_histories:
            histories = {}
            for char_name, history in ground_truth.character_histories.items():
                if hasattr(history, 'substitutions'):
                    histories[char_name] = {
                        'ancestral_states': getattr(history, 'ancestral_states', {}),
                        'final_states': getattr(history, 'final_states', {}),
                        'substitutions': getattr(history, 'substitutions', [])
                    }
            result['character_histories'] = histories
        
        return result


class OutputEngine:
    """
    Comprehensive output engine for synthetic phylogenetic datasets.
    
    Supports multiple output formats:
    - NEXUS: Compatible with PAUP*, MrBayes, BEAST, and other phylogenetic software
    - CSV: Data matrices for statistical analysis in R, Python, etc.
    - JSON: Complete metadata and ground truth for validation
    - Newick: Tree format for phylogenetic applications
    """
    
    def __init__(self, config: Any):
        """Initialize output engine with configuration."""
        self.config = config
        
        # Map format enums to writers
        self.writers = {
            OutputFormat.NEXUS: NexusWriter,
            OutputFormat.CSV: CsvWriter,
            OutputFormat.JSON: JsonWriter,
        }
        
        logger.debug(f"Output engine initialized for formats: {[f.value for f in config.formats]}")
    
    def export_dataset(self, dataset: Any, output_dir: Path) -> Dict[str, Path]:
        """
        Export dataset in all configured formats.
        
        Args:
            dataset: DatasetResult object to export
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping format names to output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = {}
        
        # Generate base filename
        base_name = self.config.base_name
        if self.config.include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{base_name}_{timestamp}"
        
        logger.info(f"Exporting dataset '{dataset.name}' to {output_dir}")
        
        # Export each configured format
        for output_format in self.config.formats:
            try:
                format_name = output_format.value
                logger.debug(f"Exporting {format_name} format")
                
                if output_format == OutputFormat.NEXUS:
                    file_path = output_dir / f"{base_name}.nexus"
                    NexusWriter.write(dataset, file_path, self.config.nexus_options)
                    output_paths[format_name] = file_path
                
                elif output_format == OutputFormat.CSV:
                    file_path = output_dir / f"{base_name}.csv"
                    csv_files = CsvWriter.write(dataset, file_path, self.config.csv_options)
                    # Return primary matrix file, but all files are created
                    output_paths[format_name] = csv_files[0] if csv_files else file_path
                
                elif output_format == OutputFormat.JSON:
                    file_path = output_dir / f"{base_name}.json"
                    JsonWriter.write(dataset, file_path, self.config.json_options)
                    output_paths[format_name] = file_path
                
                elif output_format == OutputFormat.NEWICK:
                    file_path = output_dir / f"{base_name}.tre"
                    with open(file_path, 'w') as f:
                        f.write(dataset.tree_newick + "\n")
                    output_paths[format_name] = file_path
                
                else:
                    logger.warning(f"Unsupported output format: {format_name}")
                    
            except Exception as e:
                logger.error(f"Failed to export {format_name} format: {e}")
                continue
        
        logger.info(f"Successfully exported {len(output_paths)} formats")
        return output_paths