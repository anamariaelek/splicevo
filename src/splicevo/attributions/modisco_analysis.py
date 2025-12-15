"""
TF-MoDisco analysis for discovering motifs in model attributions.

This module provides a flexible API for analyzing attributions with tfmodisco-lite,
supporting multiple input formats, filtering strategies, and visualization options.

Features:
- Support for both splice and usage attributions
- Flexible sequence selection (indices, genomic coordinates, predictions)
- Customizable aggregation strategies (per-site, per-class, per-condition)
- Integration with tfmodisco-lite for unsupervised motif discovery
- Scalable processing with batching
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import h5py
from tqdm import tqdm

try:
    import modiscolite
    from modiscolite import tfmodisco
    MODISCO_AVAILABLE = True
except ImportError:
    MODISCO_AVAILABLE = False

from splicevo.model import SplicevoModel


@dataclass
class ModiscoConfig:
    """Configuration for tfmodisco-lite analysis.
    
    Attributes:
        sliding_window_size: Core window size scanned across sequences (int)
        flank_size: Flank size added on each side during seqlet extraction (int)
        min_passing_windows_frac: Max fraction of passing windows (float),
        max_passing_windows_frac: Max fraction of passing windows (float),
        min_metacluster_size: (int)
        max_seqlets_per_metacluster: (int)
        target_seqlet_fdr: Faalse discovery rate(float) 
        trim_to_window_size: Trim seqlets to this size (int)
        initial_flank_to_add: Initial flank to add when extracting seqlets (int)
        final_flank_to_add: Final flank to add when extracting seqlets (int)
        n_processing_cores: Number of cores for processing (int)
        batch_size: Batch size for processing (int, if applicable)
    """
    sliding_window_size: int = 12
    flank_size: int = 10
    min_passing_windows_frac: float = 0.03
    max_passing_windows_frac: float = 0.2
    min_metacluster_size: float = 20
    max_seqlets_per_metacluster: int = 20000
    target_seqlet_fdr: float = 0.05
    trim_to_window_size: int = 30
    initial_flank_to_add: int = 10
    final_flank_to_add: int = 0
    n_processing_cores: int = 4
    batch_size: int = 32


@dataclass
class ModiscoInput:
    """Input data container for tfmodisco analysis.
    
    Attributes:
        attributions: Attribution arrays (n_samples, seq_len, 4) or (n_samples, seq_len, 4, n_conditions)
        sequences: One-hot encoded sequences (n_samples, seq_len, 4)
        metadata: Dictionary with sample metadata
        condition_names: Optional list of condition names
        site_types: Optional array of site types for filtering
        sites_metadata: Optional list of dictionaries with metadata for each site
    """
    attributions: np.ndarray
    sequences: np.ndarray
    metadata: Dict
    condition_names: Optional[List[str]] = None
    site_types: Optional[np.ndarray] = None
    sites_metadata: Optional[List[Dict]] = field(default_factory=list)


class AttributionAggregator:
    """Flexible aggregation of attributions for motif discovery.
    
    Supports multiple aggregation strategies:
    - by_site_type: Aggregate all donors, all acceptors separately
    - by_condition: Separate attributions by condition (usage)
    - by_prediction_correctness: Compare correct vs incorrect predictions
    - combined: Aggregate all attributions together
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize aggregator.
        
        Args:
            verbose: Print debug information
        """
        self.verbose = verbose
    

    def aggregate_by_site_type(
        self,
        attributions: np.ndarray,
        sequences: np.ndarray,
        site_types: np.ndarray,
        sites_metadata: List[Dict],
        base_metadata: Dict
    ) -> Dict[str, ModiscoInput]:
        """Aggregate attributions by splice site type using metadata from saved files.
        
        Args:
            attributions: Array of attributions (n_samples, seq_len, 4[, n_conditions])
            sequences: Array of sequences (n_samples, seq_len, 4)
            site_types: Array of site types (n_samples,) with values like 'donor', 'acceptor'
            sites_metadata: List of metadata dictionaries for each site
            base_metadata: Base metadata dictionary to include in all groups
            
        Returns:
            Dictionary mapping site_type -> ModiscoInput
        """
        if self.verbose:
            print(f"Aggregating {len(site_types)} attributions by site type with metadata")
        
        site_type_groups = {}
        for site_type in np.unique(site_types):
            mask = site_types == site_type
            
            # Filter sites metadata by mask
            filtered_sites_metadata = [sites_metadata[i] for i, m in enumerate(mask) if m]
            
            site_type_groups[site_type] = ModiscoInput(
                attributions=attributions[mask],
                sequences=sequences[mask],
                metadata={
                    **base_metadata,
                    'aggregation': 'by_site_type',
                    'site_type': site_type,
                    'n_samples': np.sum(mask)
                },
                site_types=site_types[mask],
                sites_metadata=filtered_sites_metadata
            )
            
            if self.verbose:
                print(f"  {site_type}: {np.sum(mask)} samples")
        
        return site_type_groups


    def aggregate_by_condition(
        self,
        attributions: np.ndarray,
        sequences: np.ndarray,
        condition_names: List[str],
        base_metadata: Dict,
        site_types: Optional[np.ndarray] = None,
        sites_metadata: Optional[List[Dict]] = None
    ) -> Dict[str, ModiscoInput]:
        """Aggregate usage attributions by condition using metadata from saved files.

        For multi-condition attributions expanded by save_attributions_for_modisco,
        this function separates sites by condition tracking and groups them for
        per-condition TFMoDISco analysis.

        Args:
            attributions: Array of attributions (n_samples, seq_len, 4)
                         where samples may include expanded conditions
            sequences: Array of sequences (n_samples, seq_len, 4)
            condition_names: List of condition names (e.g., ['tissue1', 'tissue2'])
            base_metadata: Base metadata dictionary to include in all groups
            site_types: Optional array of site types (n_samples,) with values like 'donor', 'acceptor'
            sites_metadata: Optional list of metadata dictionaries for each site.
                           Should contain 'condition_idx' field for expanded conditions.
            
        Returns:
            Dictionary mapping condition_name -> ModiscoInput
            
        Notes:
            - If sites_metadata contains 'condition_idx' fields, uses those for grouping
            - Otherwise, assumes attributions are already condition-separated
            - Each condition group gets its own ModiscoInput for per-condition analysis
        """
        if self.verbose:
            print(f"Aggregating {len(attributions)} attributions by condition")
        
        # If no sites_metadata provided, assume 1:1 mapping with sequences
        if sites_metadata is None:
            sites_metadata = [{'condition_idx': 0} for _ in range(len(attributions))]
        
        # Extract condition indices from sites metadata
        condition_indices = np.array([
            site.get('condition_idx', 0) for site in sites_metadata
        ])
        
        # Group by condition
        condition_groups = {}
        for cond_idx, cond_name in enumerate(condition_names):
            mask = condition_indices == cond_idx
            n_samples = np.sum(mask)
            
            if n_samples == 0:
                if self.verbose:
                    print(f"  {cond_name}: 0 samples (skipping)")
                continue
            
            # Filter sites metadata by mask
            filtered_sites_metadata = [sites_metadata[i] for i, m in enumerate(mask) if m]
            
            # Create ModiscoInput for this condition
            condition_groups[cond_name] = ModiscoInput(
                attributions=attributions[mask],
                sequences=sequences[mask],
                metadata={
                    **base_metadata,
                    'aggregation': 'by_condition',
                    'condition': cond_name,
                    'condition_idx': cond_idx,
                    'n_samples': n_samples
                },
                condition_names=[cond_name],  # Single condition for this group
                site_types=site_types[mask] if site_types is not None else None,
                sites_metadata=filtered_sites_metadata
            )
            
            if self.verbose:
                print(f"  {cond_name}: {n_samples} samples")
        
        return condition_groups
    

class ModiscoAnalyzer:
    """Main interface for tfmodisco-lite analysis on attributions.
    
    Workflow:
    1. Load attributions 
    2. Aggregate using AttributionAggregator
    3. Run tfmodisco-lite analysis
    4. Save and visualize results
    """
    
    def __init__(
        self,
        config: Optional[ModiscoConfig] = None,
        verbose: bool = False
    ):
        """Initialize analyzer.
        
        Args:
            config: ModiscoConfig instance
            verbose: Print debug information
        """
        self.config = config or ModiscoConfig()
        self.verbose = verbose
        self.aggregator = AttributionAggregator(verbose=verbose)
        self.results = {}
    
    def load_from_saved_attributions(
        self,
        base_path: Union[str, Path],
        aggregation_strategy: Literal['by_site_type', 'by_condition', 'all'] = 'all',
        filter_site_type: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Dict]:
        """Load attributions from files saved by save_attributions_for_modisco.
        
        Loads sequences, attributions, and metadata from numpy/JSON files and
        prepares them for tfmodisco analysis using the specified aggregation strategy.
        
        Args:
            base_path: Base path for saved files (without extension).
                      Will load:
                      - {base_path}_sequences.npy
                      - {base_path}_attributions.npy
                      - {base_path}_metadata.json
            aggregation_strategy: How to group attributions for analysis
                - 'by_site_type': Separate donors and acceptors
                - 'by_condition': For usage attributions, separate by condition
                - 'all': Analyze all together
            filter_site_type: Only analyze specific site type (e.g., 'donor')
            output_dir: Optional directory to save HDF5 results files
            
        Returns:
            Dictionary mapping analysis_name -> modisco result
            
        Raises:
            FileNotFoundError: If required files are not found
        """
        base_path = Path(base_path)
        
        # Load files
        sequences_path = Path(str(base_path) + "_sequences.npy")
        attributions_path = Path(str(base_path) + "_attributions.npy")
        metadata_path = Path(str(base_path) + "_metadata.json")
        
        if not sequences_path.exists():
            raise FileNotFoundError(f"Sequences file not found: {sequences_path}")
        if not attributions_path.exists():
            raise FileNotFoundError(f"Attributions file not found: {attributions_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        if self.verbose:
            print(f"Loading saved attributions from {base_path}")
        
        # Load arrays
        sequences = np.load(sequences_path)
        attributions = np.load(attributions_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_full = json.load(f)
        
        if self.verbose:
            print(f"  Sequences shape: {sequences.shape}")
            print(f"  Attributions shape: {attributions.shape}")
            print(f"  Loaded {metadata_full.get('n_sites', 'unknown')} sites")
        
        # Extract site information from metadata
        sites_metadata = metadata_full.get('sites', [])
        site_types = np.array([site.get('site_type', 'unknown') for site in sites_metadata])
        site_ids = np.array([site.get('site_id', f'site_{i}') for i, site in enumerate(sites_metadata)])
        
        # Apply filters
        if filter_site_type:
            mask = site_types == filter_site_type
            attributions = attributions[mask]
            sequences = sequences[mask]
            site_types = site_types[mask]
            sites_metadata = [sites_metadata[i] for i, m in enumerate(mask) if m]
            if self.verbose:
                print(f"  Filtered to {len(sites_metadata)} {filter_site_type} sites")
        
        # Create base metadata dictionary from the metadata JSON
        base_metadata = {
            'window': metadata_full.get('window'),
            'condition_idx': metadata_full.get('condition_idx'),
            'source': str(base_path),
            'n_sites_total': metadata_full.get('n_sites'),
        }
        
        # Aggregate based on strategy
        if aggregation_strategy == 'by_site_type':
            groups = self.aggregator.aggregate_by_site_type(
                attributions, sequences, site_types, sites_metadata, base_metadata
            )
        elif aggregation_strategy == 'by_condition':
            condition_names = metadata_full.get('condition_names', [])
            groups = self.aggregator.aggregate_by_condition(
                attributions, sequences, condition_names,
                base_metadata,
                site_types=site_types,
                sites_metadata=sites_metadata
            )
        elif aggregation_strategy == 'all':
            groups = {
                'all': ModiscoInput(
                    attributions=attributions,
                    sequences=sequences,
                    metadata={
                        **base_metadata,
                        'aggregation': 'all',
                        'n_samples': len(sequences)
                    },
                    site_types=site_types,
                    sites_metadata=sites_metadata
                )
            }
        else:
            raise ValueError(f"Aggregation strategy '{aggregation_strategy}' not yet supported for saved files. "
                           f"Use 'by_site_type', 'by_condition', or 'all'.")
        
        # Run modisco on each group
        results = {}
        for group_name, group_input in groups.items():
            if self.verbose:
                print(f"\nProcessing group: '{group_name}'")
            
            # Determine output path if output_dir provided
            output_h5 = None
            if output_dir:
                output_h5 = Path(output_dir) / f"{group_name}_motifs.h5"
            
            result = self.run_modisco(group_input, name=group_name, output_h5=output_h5)
            results[group_name] = result
        
        return results
    
    def prepare_modisco_input(
        self,
        modisco_input: ModiscoInput,
        normalize: bool = True,
        abs_value: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare attributions and sequences for tfmodisco.
        
        Args:
            modisco_input: ModiscoInput instance
            normalize: Normalize attributions to [0, 1] range
            abs_value: Use absolute values of attributions
            
        Returns:
            Tuple of (attributions, sequences) ready for tfmodisco
        """
        attributions = modisco_input.attributions.copy()
        sequences = modisco_input.sequences.copy()
        
        if self.verbose:
            print(f"  Preparing modisco input:")
            print(f"  Input shapes: attr={attributions.shape}, seq={sequences.shape}")
        
        # Handle different attribution shapes
        if len(attributions.shape) == 4:
            # Usage attributions: aggregate across conditions
            if self.verbose:
                print(f"  Aggregating across conditions")
            attributions = attributions.mean(axis=-1)  # (n_samples, seq_len, 4)
        
        if abs_value:
            attributions = np.abs(attributions)
        
        if normalize:
            # Normalize by sequence to get attribution per actual base
            # Mask non-existent bases (set to 0)
            attr_per_base = attributions * sequences
            
            # Normalize per position
            max_per_pos = np.max(np.abs(attr_per_base), axis=(0, 2), keepdims=True)
            max_per_pos[max_per_pos == 0] = 1.0  # Avoid division by zero
            attributions = attributions / max_per_pos
        
        if self.verbose:
            print(f"  Output shapes: attr={attributions.shape}, seq={sequences.shape}")
            print(f"  Attribution range: [{attributions.min():.4f}, {attributions.max():.4f}]")
        
        return attributions, sequences
    
    def run_modisco(
        self,
        modisco_input: ModiscoInput,
        name: str = 'analysis',
        output_h5: Optional[Union[str, Path]] = None
    ) -> Dict:
        """Run tfmodisco-lite on prepared input.
        
        Args:
            modisco_input: Prepared ModiscoInput instance
            name: Name for this analysis
            output_h5: Optional path to save HDF5 results file
            
        Returns:
            Dictionary with modisco results
            
        Raises:
            ImportError: If modiscolite is not installed
        """
        if not MODISCO_AVAILABLE:
            raise ImportError(
                "modiscolite is not installed. Install with: "
                "pip install modisco or pip install modisco-lite"
            )
        
        if self.verbose:
            print(f"\nRunning tfmodisco on '{name}'")
            print(f"  Samples: {modisco_input.metadata.get('n_samples', 'unknown')}")
            print(f"  Config: "
                  f"sliding_window_size={self.config.sliding_window_size},"
                  f"flank_size={self.config.flank_size},"
                  f"min_passing_windows_frac={self.config.min_passing_windows_frac},"
                  f"max_passing_windows_frac={self.config.max_passing_windows_frac},"
                  f"min_metacluster_size={self.config.min_metacluster_size},"
                  f"max_seqlets_per_metacluster={self.config.max_seqlets_per_metacluster}",
                  f"target_seqlet_fdr={self.config.target_seqlet_fdr}",
                  f"trim_to_window_size={self.config.trim_to_window_size}",
                  f"initial_flank_to_add={self.config.initial_flank_to_add}",
                  f"final_flank_to_add={self.config.final_flank_to_add}",
                  f"n_processing_cores={self.config.n_processing_cores}",
                  f"batch_size={self.config.batch_size}")
        # Prepare input
        attributions, sequences = self.prepare_modisco_input(modisco_input)
        
        # Data format: tfmodisco expects (n_samples, seq_len, 4)
        # Reshape: (n, len, 4) - current format
        if self.verbose:
            print(f"  Input shapes: attr={attributions.shape}, seq={sequences.shape}")
        
        # Run tfmodisco-lite
        try:
            if self.verbose:
                print(f"  Running TFMoDISco algorithm...")
            
            pos_patterns, neg_patterns = tfmodisco.TFMoDISco(
                one_hot=sequences,
                hypothetical_contribs=attributions,
                sliding_window_size=self.config.sliding_window_size,
                flank_size=self.config.flank_size,
                min_metacluster_size=self.config.min_metacluster_size,
                max_seqlets_per_metacluster=self.config.max_seqlets_per_metacluster,
                target_seqlet_fdr=self.config.target_seqlet_fdr,
                trim_to_window_size=self.config.trim_to_window_size,
                initial_flank_to_add=self.config.initial_flank_to_add,
                final_flank_to_add=self.config.final_flank_to_add,
                min_passing_windows_frac=self.config.min_passing_windows_frac,
                max_passing_windows_frac=self.config.max_passing_windows_frac,
                n_leiden_runs=50,
                n_leiden_iterations=-1,
            )            
            if self.verbose:
                n_pos = len(pos_patterns) if pos_patterns is not None else 0
                n_neg = len(neg_patterns) if neg_patterns is not None else 0
                print(f"    Discovered {n_pos} positive patterns and {n_neg} negative patterns")
            
        except Exception as e:
            if self.verbose:
                print(f"    Error running tfmodisco: {str(e)}")
            return {
                'name': name,
                'status': 'error',
                'error': str(e),
                'config': self.config,
                'prepared_attributions': attributions,
                'prepared_sequences': sequences,
                'metadata': modisco_input.metadata,
                'motifs': None,
            }
        
        # Save results to HDF5 if path provided
        h5_path = None
        if output_h5 is not None:
            h5_path = Path(output_h5)
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                if self.verbose:
                    print(f"  Saving results to {h5_path}...")
                modiscolite.io.save_hdf5(
                    str(h5_path),
                    pos_patterns,
                    neg_patterns,
                    window_size=400  # Standard TF-MoDISco window
                )
                if self.verbose:
                    print(f"    Results saved")
            except Exception as e:
                if self.verbose:
                    print(f"    Warning: Could not save HDF5: {str(e)}")
        
        # Create result dictionary with pattern information
        result = {
            'name': name,
            'status': 'complete',
            'config': self.config,
            'input': modisco_input,
            'prepared_attributions': attributions,
            'prepared_sequences': sequences,
            'metadata': modisco_input.metadata,
            'pos_patterns': pos_patterns,
            'neg_patterns': neg_patterns,
            'h5_path': str(h5_path) if h5_path else None,
            'motif_summary': self.summarize_patterns(pos_patterns, neg_patterns)
        }
        
        self.results[name] = result
        
        if self.verbose:
            print(f"    Result stored as '{name}'")
        
        return result
    
    def summarize_patterns(
        self,
        pos_patterns: Optional[List],
        neg_patterns: Optional[List]
    ) -> Dict:
        """Summarize discovered patterns.
        
        Args:
            pos_patterns: List of positive SeqletSet patterns
            neg_patterns: List of negative SeqletSet patterns
            
        Returns:
            Dictionary with pattern statistics
        """
        summary = {
            'n_pos_patterns': len(pos_patterns) if pos_patterns is not None else 0,
            'n_neg_patterns': len(neg_patterns) if neg_patterns is not None else 0,
            'pos_pattern_details': [],
            'neg_pattern_details': []
        }
        
        # Extract details about positive patterns
        if pos_patterns is not None:
            for idx, pattern in enumerate(pos_patterns):
                try:
                    n_seqlets = len(pattern.seqlets) if hasattr(pattern, 'seqlets') else 0
                    summary['pos_pattern_details'].append({
                        'pattern_id': idx,
                        'n_seqlets': n_seqlets,
                        'has_sequence': hasattr(pattern, 'sequence'),
                        'has_contributions': hasattr(pattern, 'contrib_scores')
                    })
                except:
                    pass
        
        # Extract details about negative patterns
        if neg_patterns is not None:
            for idx, pattern in enumerate(neg_patterns):
                try:
                    n_seqlets = len(pattern.seqlets) if hasattr(pattern, 'seqlets') else 0
                    summary['neg_pattern_details'].append({
                        'pattern_id': idx,
                        'n_seqlets': n_seqlets,
                        'has_sequence': hasattr(pattern, 'sequence'),
                        'has_contributions': hasattr(pattern, 'contrib_scores')
                    })
                except:
                    pass
        
        return summary
    
    def load_results(
        self,
        result_dir: Union[str, Path],
        name: str
    ) -> Dict:
        """Load analysis results from disk.
        
        Args:
            result_dir: Directory containing saved results
            name: Name of result to load
            
        Returns:
            Loaded result dictionary
        """
        result_dir = Path(result_dir)
        
        # Load metadata
        json_path = result_dir / f"{name}_metadata.json"
        with open(json_path, 'r') as f:
            result = json.load(f)
        
        # Load prepared data
        attr_path = result_dir / f"{name}_attributions.npy"
        seq_path = result_dir / f"{name}_sequences.npy"
        
        if attr_path.exists():
            result['prepared_attributions'] = np.load(attr_path)
        if seq_path.exists():
            result['prepared_sequences'] = np.load(seq_path)
        
        # Load HDF5 if available
        h5_path_str = result.get('h5_path')
        if h5_path_str:
            h5_path = Path(h5_path_str)
            if h5_path.exists():
                result['h5_path'] = str(h5_path)
        
        self.results[name] = result
        
        if self.verbose:
            print(f"Loaded result '{name}' from {result_dir}")
        
        return result
    
    def extract_motifs_from_h5(
        self,
        h5_path: Union[str, Path]
    ) -> Dict[str, Dict]:
        """Extract motif information from HDF5 results file.
        
        Args:
            h5_path: Path to modisco results HDF5 file
            
        Returns:
            Dictionary with extracted motif information
        """
        if not MODISCO_AVAILABLE:
            raise ImportError("modiscolite is not installed")
        
        h5_path = Path(h5_path)
        motifs = {'pos_patterns': [], 'neg_patterns': []}
        
        try:
            with h5py.File(h5_path, 'r') as f:
                for pattern_group_name in ['pos_patterns', 'neg_patterns']:
                    if pattern_group_name not in f:
                        continue
                    
                    pattern_group = f[pattern_group_name]
                    
                    for pattern_name in sorted(pattern_group.keys()):
                        pattern = pattern_group[pattern_name]
                        
                        # Extract pattern information
                        motif_info = {
                            'name': pattern_name,
                            'sequence': np.array(pattern['sequence'][:]) if 'sequence' in pattern else None,
                            'contrib_scores': np.array(pattern['contrib_scores'][:]) if 'contrib_scores' in pattern else None,
                            'hypothetical_contribs': np.array(pattern['hypothetical_contribs'][:]) if 'hypothetical_contribs' in pattern else None,
                        }
                        
                        # Extract seqlet information if available
                        if 'seqlets' in pattern:
                            seqlets = pattern['seqlets']
                            motif_info['n_seqlets'] = int(seqlets['n_seqlets'][()])
                            motif_info['seqlet_positions'] = {
                                'start': np.array(seqlets['start'][:]),
                                'end': np.array(seqlets['end'][:]),
                                'example_idx': np.array(seqlets['example_idx'][:]),
                                'is_revcomp': np.array(seqlets['is_revcomp'][:]),
                            }
                        
                        motifs[pattern_group_name].append(motif_info)
            
            if self.verbose:
                print(f"Extracted {len(motifs['pos_patterns'])} positive and "
                      f"{len(motifs['neg_patterns'])} negative motifs from {h5_path}")
        
        except Exception as e:
            if self.verbose:
                print(f"Error extracting motifs: {str(e)}")
        
        return motifs
    
    def generate_reports(
        self,
        h5_path: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> Path:
        """Generate HTML reports from modisco results using modiscolite.
        
        Args:
            h5_path: Path to modisco results HDF5 file
            output_dir: Directory to save reports
            
        Returns:
            Path to generated reports
            
        Note:
            Requires modiscolite with reporting capability.
            Optional: provide meme_motif_db for TOMTOM comparison.
        """
        if not MODISCO_AVAILABLE:
            raise ImportError("modiscolite is not installed")
        
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if self.verbose:
                print(f"Generating reports in {output_dir}...")
            
            # Try to use modiscolite's report generation
            try:
                from modiscolite import report
                report.report_motifs(
                    modisco_h5py=str(h5_path),
                    output_dir=str(output_dir),
                    img_path_suffix="./",
                    meme_motif_db=None,
                    is_writing_tomtom_matrix=False,
                )
                if self.verbose:
                    print(f"  Reports generated in {output_dir}")
            except Exception as e:
                if self.verbose:
                    print(f"  Could not generate full report: {str(e)}")
                print(f"Note: You can generate reports manually using:")
                print(f"  modisco report -i {h5_path} -o {output_dir}")
        
        except Exception as e:
            if self.verbose:
                print(f"Error: {str(e)}")
        
        return output_dir
    
    def get_summary(self) -> Dict:
        """Get summary of all analyses performed.
        
        Returns:
            Dictionary with analysis summaries
        """
        summary = {}
        for name, result in self.results.items():
            summary[name] = {
                'status': result.get('status', 'unknown'),
                'n_samples': result.get('metadata', {}).get('n_samples', 'unknown'),
                'aggregation': result.get('metadata', {}).get('aggregation', 'unknown'),
                'has_motifs': result.get('motifs') is not None
            }
        return summary


# Convenience functions for quick analysis

def analyze_saved_attributions_quick(
    base_path: Union[str, Path],
    aggregation: Literal['by_site_type', 'all'] = 'all',
    filter_site_type: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[ModiscoConfig] = None,
    verbose: bool = True
) -> ModiscoAnalyzer:
    """Quick start for modisco analysis on saved attribution files.
    
    Load attributions from files saved by save_attributions_for_modisco and
    perform tfmodisco analysis.
    
    Args:
        base_path: Base path for saved files (without extension).
                  Will load:
                  - {base_path}_sequences.npy
                  - {base_path}_attributions.npy
                  - {base_path}_metadata.json
        aggregation: Aggregation strategy ('by_site_type' or 'combined')
        filter_site_type: Optional filter for specific site type (e.g., 'donor')
        output_dir: Optional directory to save HDF5 results files
        config: Optional ModiscoConfig instance
        verbose: Print progress information
        
    Returns:
        ModiscoAnalyzer instance with completed analysis
        
    Example:
        >>> analyzer = analyze_saved_attributions_quick(
        ...     'modisco_data/splice_attributions',
        ...     aggregation='by_site_type',
        ...     output_dir='modisco_results/'
        ... )
    """
    analyzer = ModiscoAnalyzer(config=config, verbose=verbose)
    analyzer.load_from_saved_attributions(
        base_path,
        aggregation_strategy=aggregation,
        filter_site_type=filter_site_type,
        output_dir=output_dir
    )
    
    return analyzer
