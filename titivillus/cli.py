#!/usr/bin/env python3
"""
Command-line interface for Titivillus synthetic phylogenetic data generator.

This module provides the main CLI entry point with commands for generating
datasets, validating configurations, and managing templates.
"""

import click
import sys
from pathlib import Path
import logging
from typing import Optional
import yaml

from .generator import Generator
from .config import Config
from .templates import TemplateManager


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.version_option(version='0.1.0', prog_name='titivillus')
def main(verbose: bool = False):
    """
    Titivillus: Synthetic phylogenetic data generator.
    
    Named after the demon of scribal errors, Titivillus generates realistic
    synthetic phylogenetic datasets with comprehensive ground truth for
    algorithm testing and validation.
    
    Examples:
        # Generate dataset from configuration
        titivillus generate --config my_config.yaml --output ./results
        
        # Quick linguistic dataset
        titivillus generate --template linguistic --taxa 20 --characters 50 --output ./data
        
        # Validate configuration
        titivillus validate --config my_config.yaml
        
        # List available templates
        titivillus templates
    """
    setup_logging(verbose)


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to YAML configuration file')
@click.option('--template', '-t', type=str,
              help='Use predefined template (linguistic, stemmatology, cultural, multidomain)')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output directory for generated files')
@click.option('--name', '-n', type=str,
              help='Name for the dataset (overrides config)')
@click.option('--taxa', type=int, help='Number of taxa (for templates)')
@click.option('--characters', type=int, help='Number of characters (for templates)')
@click.option('--seed', type=int, help='Random seed (overrides config)')
@click.option('--dry-run', is_flag=True, 
              help='Validate configuration without generating data')
def generate(config: Optional[str], template: Optional[str], output: str, 
             name: Optional[str], taxa: Optional[int], characters: Optional[int], 
             seed: Optional[int], dry_run: bool):
    """Generate synthetic phylogenetic dataset."""
    
    try:
        # Create generator from config or template
        if config:
            click.echo(f"Loading configuration from: {config}")
            generator = Generator.from_yaml(config)
            
            # Override config values if provided
            if name:
                generator.config.name = name
            if taxa:
                generator.config.tree.taxa_count = taxa
            if characters:
                # Distribute characters across active domains
                active_domains = generator.config.active_domains
                chars_per_domain = characters // len(active_domains)
                for domain in active_domains:
                    generator.config.domains[domain].character_count = chars_per_domain
            if seed:
                generator.config.seed = seed
                generator.rng = generator.rng.__class__(seed)
                
        elif template:
            click.echo(f"Using template: {template}")
            
            # Set defaults
            taxa = taxa or 20
            characters = characters or 50
            seed = seed or 42
            dataset_name = name or f"{template}_dataset"
            
            # Create generator from template
            if template == 'linguistic':
                generator = Generator.quick_linguistic(taxa, characters, seed)
            elif template == 'stemmatology':
                generator = Generator.quick_stemmatology(taxa, characters, seed)
            elif template == 'multidomain':
                generator = Generator.quick_multidomain(taxa, seed)
            else:
                # Try to load from template manager
                template_manager = TemplateManager()
                if template_manager.has_template(template):
                    template_config = template_manager.get_template(template)
                    # Apply overrides
                    if taxa:
                        template_config.tree.taxa_count = taxa
                    if characters:
                        active_domains = template_config.active_domains
                        chars_per_domain = characters // len(active_domains)
                        for domain in active_domains:
                            template_config.domains[domain].character_count = chars_per_domain
                    if seed:
                        template_config.seed = seed
                    if name:
                        template_config.name = name
                        
                    generator = Generator(template_config)
                else:
                    click.echo(f"Error: Unknown template '{template}'", err=True)
                    click.echo("Available templates: linguistic, stemmatology, multidomain", err=True)
                    sys.exit(1)
                    
            if name:
                generator.config.name = dataset_name
                
        else:
            click.echo("Error: Must specify either --config or --template", err=True)
            sys.exit(1)
        
        # Validate configuration
        is_valid, issues = generator.validate_config()
        if not is_valid:
            click.echo("Configuration validation failed:", err=True)
            for issue in issues:
                click.echo(f"  - {issue}", err=True)
            sys.exit(1)
        
        click.echo(f"Configuration valid: {generator.config.name}")
        
        # Show generation plan
        click.echo("\nGeneration Plan:")
        click.echo(f"  Dataset: {generator.config.name}")
        click.echo(f"  Taxa: {generator.config.tree.taxa_count}")
        
        total_chars = sum(
            generator.config.domains[domain].character_count 
            for domain in generator.config.active_domains
        )
        click.echo(f"  Characters: {total_chars}")
        click.echo(f"  Domains: {[d.value for d in generator.config.active_domains]}")
        click.echo(f"  Output formats: {[f.value for f in generator.config.output.formats]}")
        
        if generator.config.networks.enable_reticulation:
            click.echo(f"  Network effects: enabled (rate={generator.config.networks.reticulation_rate})")
        else:
            click.echo("  Network effects: disabled")
            
        click.echo(f"  Error injection: {generator.config.errors.missing_data_rate:.1%} missing data")
        click.echo(f"  Random seed: {generator.config.seed}")
        
        if dry_run:
            click.echo("\nDry run completed - configuration is valid")
            return
        
        # Generate dataset
        click.echo("\nGenerating dataset...")
        with click.progressbar(length=100, label='Generating') as bar:
            # This is a simplified progress bar - in reality we'd need
            # progress callbacks from the generator
            dataset, output_paths = generator.generate_and_export(output, name)
            bar.update(100)
        
        # Report results
        click.echo(f"\n✅ Dataset generated successfully!")
        click.echo(f"Name: {dataset.name}")
        click.echo(f"Taxa: {len(dataset.taxa)}")
        click.echo(f"Characters: {len(dataset.characters)}")
        click.echo(f"Generation time: {dataset.stats['generation_time']:.2f}s")
        
        click.echo(f"\nOutput files ({len(output_paths)}):")
        for format_name, file_path in output_paths.items():
            if isinstance(file_path, Path):
                size_kb = file_path.stat().st_size / 1024 if file_path.exists() else 0
                click.echo(f"  {format_name}: {file_path.name} ({size_kb:.1f} KB)")
            else:
                click.echo(f"  {format_name}: {file_path}")
        
        click.echo(f"\nAll files saved to: {Path(output).absolute()}")
        
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error generating dataset: {e}", err=True)
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True,
              help='Path to YAML configuration file to validate')
@click.option('--strict', is_flag=True, 
              help='Use strict validation (fail on warnings)')
def validate(config: str, strict: bool):
    """Validate a configuration file."""
    try:
        click.echo(f"Validating configuration: {config}")
        
        # Load and validate config
        config_obj = Config.from_yaml(config)
        issues = config_obj.validate()
        
        if not issues:
            click.echo("✅ Configuration is valid")
            
            # Show configuration summary
            click.echo("\nConfiguration Summary:")
            click.echo(f"  Name: {config_obj.name}")
            click.echo(f"  Taxa: {config_obj.tree.taxa_count}")
            
            total_chars = sum(
                config_obj.domains[domain].character_count 
                for domain in config_obj.active_domains
            )
            click.echo(f"  Total characters: {total_chars}")
            click.echo(f"  Active domains: {[d.value for d in config_obj.active_domains]}")
            click.echo(f"  Output formats: {[f.value for f in config_obj.output.formats]}")
            click.echo(f"  Random seed: {config_obj.seed}")
            
        else:
            click.echo("❌ Configuration has issues:", err=True)
            for issue in issues:
                click.echo(f"  - {issue}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error validating configuration: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--list', '-l', 'list_templates', is_flag=True,
              help='List all available templates')
@click.option('--show', '-s', type=str,
              help='Show details for specific template')
@click.option('--output', '-o', type=click.Path(),
              help='Export template to file')
def templates(list_templates: bool, show: Optional[str], output: Optional[str]):
    """Manage configuration templates."""
    
    template_manager = TemplateManager()
    
    if list_templates:
        click.echo("Available templates:")
        templates_list = template_manager.list_templates()
        
        for template_name in sorted(templates_list):
            template_info = template_manager.get_template_info(template_name)
            click.echo(f"  {template_name}: {template_info['description']}")
        
        if not templates_list:
            click.echo("  No templates found")
            
        # Also show quick templates
        click.echo("\nQuick templates (built-in):")
        click.echo("  linguistic: Quick linguistic dataset generation")
        click.echo("  stemmatology: Quick manuscript tradition analysis")  
        click.echo("  multidomain: Multi-domain dataset with all domains")
        
    elif show:
        try:
            if show in ['linguistic', 'stemmatology', 'multidomain']:
                click.echo(f"Quick template: {show}")
                click.echo("This is a built-in template. Use with:")
                click.echo(f"  titivillus generate --template {show} --taxa 20 --characters 50 --output ./data")
            else:
                template_config = template_manager.get_template(show)
                click.echo(f"Template: {show}")
                click.echo(f"Description: {template_config.description}")
                click.echo(f"Taxa: {template_config.tree.taxa_count}")
                
                total_chars = sum(
                    template_config.domains[domain].character_count 
                    for domain in template_config.active_domains
                )
                click.echo(f"Characters: {total_chars}")
                click.echo(f"Domains: {[d.value for d in template_config.active_domains]}")
                
                if output:
                    template_config.to_yaml(output)
                    click.echo(f"Template saved to: {output}")
                    
        except ValueError as e:
            click.echo(f"Error: Template '{show}' not found", err=True)
            sys.exit(1)
            
    else:
        # Show general template information
        click.echo("Template Management")
        click.echo("Use --list to see available templates")
        click.echo("Use --show TEMPLATE to see template details") 
        click.echo("Use --output FILE with --show to export template")


@main.command()
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output file for example configuration')
@click.option('--template', '-t', type=str, default='basic',
              help='Type of example (basic, linguistic, stemmatology, multidomain)')
def init(output: str, template: str):
    """Create an example configuration file."""
    
    try:
        output_path = Path(output)
        
        if output_path.exists():
            if not click.confirm(f"File {output} already exists. Overwrite?"):
                click.echo("Cancelled")
                return
        
        # Create appropriate example configuration
        if template == 'basic':
            config = Config(
                name="example_dataset",
                description="Example synthetic phylogenetic dataset",
                seed=12345
            )
        elif template == 'linguistic':
            generator = Generator.quick_linguistic()
            config = generator.config
            config.description = "Example linguistic phylogenetic dataset"
        elif template == 'stemmatology':
            generator = Generator.quick_stemmatology()
            config = generator.config
            config.description = "Example manuscript tradition dataset"
        elif template == 'multidomain':
            generator = Generator.quick_multidomain()
            config = generator.config
            config.description = "Example multi-domain phylogenetic dataset"
        else:
            click.echo(f"Error: Unknown template type '{template}'", err=True)
            click.echo("Available types: basic, linguistic, stemmatology, multidomain", err=True)
            sys.exit(1)
        
        # Save configuration
        config.to_yaml(output_path)
        
        click.echo(f"✅ Example configuration created: {output_path}")
        click.echo("Edit the file to customize your dataset, then run:")
        click.echo(f"  titivillus generate --config {output} --output ./data")
        
    except Exception as e:
        click.echo(f"Error creating configuration: {e}", err=True)
        sys.exit(1)


@main.command()  
def info():
    """Show information about Titivillus."""
    click.echo("Titivillus - Synthetic Phylogenetic Data Generator")
    click.echo("=" * 50)
    click.echo()
    click.echo("Named after the demon of scribal errors, Titivillus generates")
    click.echo("realistic synthetic phylogenetic datasets with comprehensive")
    click.echo("ground truth for rigorous algorithm testing and validation.")
    click.echo()
    click.echo("Features:")
    click.echo("  • Multi-domain support (linguistics, stemmatology, cultural evolution)")
    click.echo("  • Realistic evolutionary modeling with asymmetric processes")
    click.echo("  • Network effects: reticulation, borrowing, contact zones")
    click.echo("  • Comprehensive error injection and missing data patterns")
    click.echo("  • Ground truth tracking for algorithm validation")
    click.echo("  • Multiple output formats (NEXUS, CSV, JSON, Newick)")
    click.echo("  • YAML-based configuration system")
    click.echo()
    click.echo("Quick start:")
    click.echo("  titivillus init --output config.yaml --template linguistic")
    click.echo("  titivillus generate --config config.yaml --output ./data")
    click.echo()
    click.echo("For more information, see: https://github.com/titivillus/titivillus")


if __name__ == '__main__':
    main()