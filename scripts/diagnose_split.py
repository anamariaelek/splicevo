"""Diagnose train/test split issues."""

import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, Set


def load_orthology_groups(orthology_file: str) -> Dict[str, Set[str]]:
    """Load orthology groups from TSV file."""
    orthology = {}
    with open(orthology_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            members = line.split('\t')
            group_id = f"group_{len(orthology)}"
            orthology[group_id] = set(members)
    return orthology


def diagnose(
    input_dir: str,
    orthology_file: str,
    pov_genome: str,
    test_chromosomes: list,
):
    """Diagnose the split."""
    input_path = Path(input_dir)
    test_chrom_set = set(str(c) for c in test_chromosomes)
    
    # Load orthology
    print(f"Loading orthology from {orthology_file}...")
    orthology = load_orthology_groups(orthology_file)
    print(f"  Found {len(orthology)} orthology groups")
    
    gene_to_group = {}
    for group_id, members in orthology.items():
        for member in members:
            gene_to_group[member] = group_id
    
    # Load POV genome
    print(f"\nLoading POV genome {pov_genome}...")
    pov_metadata = pd.read_csv(input_path / pov_genome / "metadata.csv")
    pov_metadata['chromosome'] = pov_metadata['chromosome'].astype(str)
    
    pov_metadata['orthology_group'] = pov_metadata['gene_id'].map(gene_to_group)
    
    test_mask = pov_metadata['chromosome'].isin(test_chrom_set)
    test_genes = set(pov_metadata[test_mask]['gene_id'].unique())
    train_genes = set(pov_metadata[~test_mask]['gene_id'].unique())
    test_groups = set(pov_metadata[test_mask]['orthology_group'].dropna().unique())
    train_groups = set(pov_metadata[~test_mask]['orthology_group'].dropna().unique())
    
    print(f"  Test genes (chr {sorted(test_chrom_set)}): {len(test_genes)}")
    print(f"  Train genes: {len(train_genes)}")
    print(f"  Test orthology groups: {len(test_groups)}")
    print(f"  Train orthology groups: {len(train_groups)}")
    print(f"  Groups only in test: {len(test_groups - train_groups)}")
    print(f"  Groups in both: {len(test_groups & train_groups)}")
    
    # Check each genome
    print(f"\nAnalyzing genomes...")
    for genome_dir in sorted(input_path.glob("*/")):
        if not genome_dir.is_dir():
            continue
        genome = genome_dir.name
        
        metadata_file = genome_dir / "metadata.csv"
        if not metadata_file.exists():
            continue
            
        metadata = pd.read_csv(metadata_file)
        metadata['orthology_group'] = metadata['gene_id'].map(gene_to_group)
        
        # Count genes in each split
        genes_in_train_groups = metadata[
            metadata['orthology_group'].isin(train_groups)
        ]
        genes_in_test_groups = metadata[
            metadata['orthology_group'].isin(test_groups)
        ]
        genes_in_both_groups = metadata[
            metadata['orthology_group'].isin(test_groups & train_groups)
        ]
        genes_unmapped = metadata[metadata['orthology_group'].isna()]
        
        print(f"\n  {genome}:")
        print(f"    Total rows: {len(metadata)}")
        print(f"    Genes in train-only groups: {len(genes_in_train_groups)}")
        print(f"    Genes in test-only groups: {len(genes_in_test_groups)}")
        print(f"    Genes in both groups: {len(genes_in_both_groups)}")
        print(f"    Unmapped genes: {len(genes_unmapped)}")
        
        if genome == pov_genome:
            print(f"    (This is POV genome - using chromosome split)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--orthology_file", required=True)
    parser.add_argument("--pov_genome", required=True)
    parser.add_argument("--test_chromosomes", type=int, nargs="+", required=True)
    
    args = parser.parse_args()
    diagnose(
        args.input_dir,
        args.orthology_file,
        args.pov_genome,
        args.test_chromosomes,
    )
