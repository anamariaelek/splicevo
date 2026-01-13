#!/usr/bin/env python3
"""
Merge attribution results from multiple batches into single files.

This script combines attribution files that were processed in batches to avoid OOM issues.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import glob


def merge_attribution_batches(input_dir, output_dir=None, verbose=True):
    """
    Merge attribution batches from separate directories.
    
    Args:
        input_dir: Directory containing batch_*/ subdirectories
        output_dir: Output directory (default: same as input_dir)
        verbose: Print progress information
    """
    input_path = Path(input_dir)
    
    # Find all batch directories
    batch_dirs = sorted(input_path.glob("batch_*"))
    
    if len(batch_dirs) == 0:
        print(f"No batch directories found in {input_dir}")
        return 1
    
    if verbose:
        print(f"Found {len(batch_dirs)} batch directories")
    
    # Create output directory
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check which attribution types exist (splice and/or usage)
    attr_types = set()
    for batch_dir in batch_dirs:
        if (batch_dir / "splice_attributions.npy").exists():
            attr_types.add("splice")
        if (batch_dir / "usage_attributions.npy").exists():
            attr_types.add("usage")
    
    if verbose:
        print(f"Attribution types found: {attr_types}")
    
    # Merge each attribution type
    for attr_type in attr_types:
        if verbose:
            print(f"\nMerging {attr_type} attributions...")
        
        # First pass: count total sites and get shapes
        total_sites = 0
        seq_shape = None
        attr_shape = None
        all_metadata = []
        
        for batch_dir in batch_dirs:
            if not batch_dir.exists():
                continue
            
            seq_file = batch_dir / f"{attr_type}_sequences.npy"
            attr_file = batch_dir / f"{attr_type}_attributions.npy"
            meta_file = batch_dir / f"{attr_type}_metadata.json"
            
            if not seq_file.exists() or not attr_file.exists():
                if verbose:
                    print(f"  Skipping {batch_dir.name}: missing files")
                continue
            
            # Load just to get shape (mmap mode doesn't load into memory)
            sequences = np.load(str(seq_file), mmap_mode='r')
            attributions = np.load(str(attr_file), mmap_mode='r')
            
            if seq_shape is None:
                seq_shape = sequences.shape[1:]  # All dims except first (batch)
                attr_shape = attributions.shape[1:]
            
            total_sites += sequences.shape[0]
            
            # Collect metadata
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                    if isinstance(metadata, list):
                        all_metadata.extend(metadata)
                    elif isinstance(metadata, dict) and 'sites' in metadata:
                        all_metadata.extend(metadata['sites'])
            
            if verbose:
                print(f"  {batch_dir.name}: {sequences.shape[0]} sites")
        
        if total_sites == 0:
            if verbose:
                print(f"No {attr_type} data found")
            continue
        
        if verbose:
            print(f"\n  Total sites to merge: {total_sites}")
            print(f"  Sequence shape per site: {seq_shape}")
            print(f"  Attribution shape per site: {attr_shape}")
        
        # Create output directory
        output_subdir = output_path / attr_type
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Pre-allocate output arrays
        merged_seq_path = output_subdir / f"{attr_type}_sequences.npy"
        merged_attr_path = output_subdir / f"{attr_type}_attributions.npy"
        
        merged_sequences = np.lib.format.open_memmap(
            str(merged_seq_path), mode='w+', 
            dtype=np.float32, shape=(total_sites,) + seq_shape
        )
        merged_attributions = np.lib.format.open_memmap(
            str(merged_attr_path), mode='w+',
            dtype=np.float32, shape=(total_sites,) + attr_shape
        )
        
        # Second pass: copy data batch by batch
        offset = 0
        for batch_dir in batch_dirs:
            if not batch_dir.exists():
                continue
            
            seq_file = batch_dir / f"{attr_type}_sequences.npy"
            attr_file = batch_dir / f"{attr_type}_attributions.npy"
            
            if not seq_file.exists() or not attr_file.exists():
                continue
            
            # Load batch (mmap mode for memory efficiency)
            sequences = np.load(str(seq_file), mmap_mode='r')
            attributions = np.load(str(attr_file), mmap_mode='r')
            
            batch_size = sequences.shape[0]
            
            # Copy to output (in chunks to avoid memory spikes)
            merged_sequences[offset:offset+batch_size] = sequences[:]
            merged_attributions[offset:offset+batch_size] = attributions[:]
            
            offset += batch_size
            
            if verbose and offset % 1000000 == 0:
                print(f"  Progress: {offset}/{total_sites} sites copied")
        
        # Flush to disk
        del merged_sequences
        del merged_attributions
        
        # Save metadata
        if all_metadata:
            with open(output_subdir / f"{attr_type}_metadata.json", 'w') as f:
                json.dump(all_metadata, f, indent=2)
        
        if verbose:
            print(f"  ✓ Merged {total_sites} total sites")
            print(f"  Saved to: {output_subdir}")
    
    if verbose:
        print(f"\nMerged results saved to: {output_path}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Merge attribution results from multiple batches"
    )
    parser.add_argument(
        '--input', required=True,
        help='Input directory containing batch_*/ subdirectories'
    )
    parser.add_argument(
        '--output', default=None,
        help='Output directory (default: same as input_dir)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    try:
        return merge_attribution_batches(
            args.input,
            args.output,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
