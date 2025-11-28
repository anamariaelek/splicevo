"""Data splitting utilities for splice site model training."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """Splits data into train/test sets based on chromosomes and orthology groups."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        orthology_file: str,
        pov_genome: str,
        test_chromosomes: List[int],
        quiet: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.orthology_file = Path(orthology_file)
        self.pov_genome = pov_genome
        self.test_chromosomes = set(str(c) for c in test_chromosomes)
        self.quiet = quiet

    def load_metadata(self, genome: str) -> pd.DataFrame:
        """Load metadata CSV for a genome."""
        metadata_path = self.input_dir / genome / "metadata.csv"
        return pd.read_csv(metadata_path)

    def load_orthology_groups(self) -> Dict[str, Set[str]]:
        """Load orthology groups from TSV file."""
        orthology = {}
        with open(self.orthology_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                members = line.split('\t')
                group_id = f"group_{len(orthology)}"
                orthology[group_id] = set(members)
        return orthology

    def get_gene_to_orthology_group(
        self, orthology_groups: Dict[str, Set[str]]
    ) -> Dict[str, str]:
        """Map each gene to its orthology group."""
        gene_to_group = {}
        for group_id, members in orthology_groups.items():
            for member in members:
                gene_to_group[member] = group_id
        return gene_to_group

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Returns:
            Tuple of (train_data, test_data) DataFrames
        """
        # Load POV genome metadata
        pov_metadata = self.load_metadata(self.pov_genome)
        
        # Ensure chromosome is string for consistent comparison
        pov_metadata['chromosome'] = pov_metadata['chromosome'].astype(str)
        
        # Split by chromosome
        test_mask = pov_metadata['chromosome'].isin(self.test_chromosomes)
        test_data = pov_metadata[test_mask].copy()
        train_data = pov_metadata[~test_mask].copy()
        
        # Load orthology groups
        orthology_groups = self.load_orthology_groups()
        gene_to_group = self.get_gene_to_orthology_group(orthology_groups)
        
        # Map genes to orthology groups
        test_data['orthology_group'] = test_data['gene_id'].map(gene_to_group)
        train_data['orthology_group'] = train_data['gene_id'].map(gene_to_group)
        
        # Log statistics
        if not self.quiet:
            test_genes = test_data['gene_id'].nunique()
            train_genes = train_data['gene_id'].nunique()
            test_groups = test_data['orthology_group'].nunique()
            train_groups = train_data['orthology_group'].nunique()
            
            logger.info(f"  POV genome test genes (chrom {sorted(self.test_chromosomes)}): {test_genes}")
            logger.info(f"  POV genome train genes: {train_genes}")
            logger.info(f"  Test ortholog groups: {test_groups}")
            logger.info(f"  Train ortholog groups: {train_groups}")
            logger.info(f"  Total gene assignments: {len(pov_metadata)}")
            logger.info(f"    Train: {len(train_data)}")
            logger.info(f"    Test: {len(test_data)}")
        
        return train_data, test_data

    def save_splits(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Save train and test splits to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = self.output_dir / "train_metadata.csv"
        test_path = self.output_dir / "test_metadata.csv"
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        if not self.quiet:
            logger.info(f"Saved train split: {train_path}")
            logger.info(f"Saved test split: {test_path}")