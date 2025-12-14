#!/usr/bin/env python3
"""
Compute and save attributions for all splice sites in the dataset.

This script performs:
1. Load model and data
2. Compute attributions for all splice sites (correctly predicted only)
3. Save attributions to disk in a reusable format

Usage:
    python compute_attributions.py \
        --model MODEL_PATH \
        --data DATA_PATH \
        --predictions PREDICTIONS_PATH \
        --output OUTPUT_DIR
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from splicevo.utils.data_utils import load_processed_data, load_predictions
from splicevo.utils.model_utils import load_model_and_config
from splicevo.attributions.compute import compute_attributions_splice


def setup_paths(model_path, data_path, predictions_path, output_dir):
    """Verify and prepare paths."""
    model_path = Path(model_path)
    data_path = Path(data_path)
    predictions_path = Path(predictions_path)
    output_dir = Path(output_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Output: {output_dir}")
    
    return model_path, data_path, predictions_path, output_dir


def load_model_and_data(model_path, data_path, predictions_path, device='cuda'):
    """Load model, sequences, labels, and metadata."""
    
    print("Loading model...")
    model, model_config = load_model_and_config(str(model_path), device=device)
    print(f"  Model loaded (config keys: {list(model_config.keys())[:3]}...)")
    
    print("Loading sequences and labels...")
    sequences, labels, _, _, usage, _ = load_processed_data(str(data_path), verbose=False)
    metadata = pd.read_csv(str(data_path / "metadata.csv"))
    print(f"  Sequences: {sequences.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Usage: {usage.shape}")
    print(f"  Metadata: {metadata.shape}")
    
    print("Loading predictions...")
    pred_preds, pred_probs, pred_sse, meta, _, _ = load_predictions(
        str(predictions_path), verbose=False
    )
    print(f"  Predictions: {pred_preds.shape}")
    
    print("Loading condition names...")
    metadata_path = predictions_path / "metadata.json"
    with open(metadata_path, 'r') as f:
        meta_json = json.load(f)
        condition_names = meta_json.get('conditions', None)
    print(f"  Conditions: {len(condition_names)}")
    
    return model, model_config, sequences, labels, usage, metadata, \
           pred_preds, pred_probs, pred_sse, condition_names


def compute_attributions(model, sequences, labels, metadata, window_indices, pred_preds, device='cuda'):
    """Compute attributions for all splice sites."""
    
    result_splice = compute_attributions_splice(
        model, sequences, labels, metadata,
        window_indices=window_indices,
        predictions=pred_preds,
        filter_by_correct=True,
        device=device,
        verbose=True
    )
    
    # Summary
    n_total = len(result_splice['attributions'])
    n_donor = sum(1 for v in result_splice['attributions'].values() 
                  if v['site_type'] == 'donor')
    n_acceptor = sum(1 for v in result_splice['attributions'].values() 
                     if v['site_type'] == 'acceptor')
    
    print(f"\nComputed attributions for {n_total} splice sites:")
    print(f"  - Donor sites: {n_donor}")
    print(f"  - Acceptor sites: {n_acceptor}")
    
    # Show sample
    print(f"\nSample attributions:")
    for i, (site_id, attr_data) in enumerate(result_splice['attributions'].items()):
        if i >= 3:
            break
        print(f"  {site_id}: {attr_data['site_type']:8s} {attr_data['attribution'].shape}")
    
    return result_splice


def save_attributions(result_splice, output_dir):
    """Save attribution results to disk in a reusable format."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate donor and acceptor attributions
    donor_attrs = {}
    acceptor_attrs = {}
    
    for site_id, attr_data in result_splice['attributions'].items():
        if attr_data['site_type'] == 'donor':
            donor_attrs[site_id] = attr_data
        else:
            acceptor_attrs[site_id] = attr_data
    
    # Save donor attributions
    donor_file = output_dir / "donor_attributions.npz"
    donor_data = {}
    donor_metadata = {}
    for site_id, attr_data in donor_attrs.items():
        donor_data[f"{site_id}_attribution"] = attr_data['attribution']
        donor_metadata[site_id] = {
            'sequence_idx': int(attr_data.get('sequence_idx', -1)),
            'position': int(attr_data.get('position', -1)),
            'site_type': 'donor'
        }
    
    np.savez_compressed(donor_file, **donor_data)
    with open(output_dir / "donor_metadata.json", 'w') as f:
        json.dump(donor_metadata, f, indent=2)
    print(f"Saved {len(donor_attrs)} donor attributions to: {donor_file}")
    
    # Save acceptor attributions
    acceptor_file = output_dir / "acceptor_attributions.npz"
    acceptor_data = {}
    acceptor_metadata = {}
    for site_id, attr_data in acceptor_attrs.items():
        acceptor_data[f"{site_id}_attribution"] = attr_data['attribution']
        acceptor_metadata[site_id] = {
            'sequence_idx': int(attr_data.get('sequence_idx', -1)),
            'position': int(attr_data.get('position', -1)),
            'site_type': 'acceptor'
        }
    
    np.savez_compressed(acceptor_file, **acceptor_data)
    with open(output_dir / "acceptor_metadata.json", 'w') as f:
        json.dump(acceptor_metadata, f, indent=2)
    print(f"Saved {len(acceptor_attrs)} acceptor attributions to: {acceptor_file}")
    
    # Save complete result dict (for TF-MoDISco)
    result_file = output_dir / "result_splice.npz"
    result_data = {}
    result_metadata = {}
    
    for site_id, attr_data in result_splice['attributions'].items():
        result_data[f"{site_id}_attribution"] = attr_data['attribution']
        result_metadata[site_id] = {
            'sequence_idx': int(attr_data.get('sequence_idx', -1)),
            'position': int(attr_data.get('position', -1)),
            'site_type': attr_data['site_type']
        }
    
    np.savez_compressed(result_file, **result_data)
    with open(output_dir / "result_metadata.json", 'w') as f:
        json.dump(result_metadata, f, indent=2)
    print(f"Saved complete result dict to: {result_file}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_sites': len(result_splice['attributions']),
        'donor_sites': len(donor_attrs),
        'acceptor_sites': len(acceptor_attrs),
        'attribution_shape': str(list(result_splice['attributions'].values())[0]['attribution'].shape)
    }
    
    summary_file = output_dir / "attribution_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_file}")
    
    return {
        'donor_file': str(donor_file),
        'acceptor_file': str(acceptor_file),
        'result_file': str(result_file),
        'donor_metadata': str(output_dir / "donor_metadata.json"),
        'acceptor_metadata': str(output_dir / "acceptor_metadata.json"),
        'result_metadata': str(output_dir / "result_metadata.json"),
        'summary': summary
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute and save attributions for all splice sites"
    )
    parser.add_argument(
        '--model', required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--data', required=True,
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--predictions', required=True,
        help='Path to predictions directory'
    )
    parser.add_argument(
        '--windows', required=False,
        help='Indices of sequences to compute attributions for (optional). ' \
        'It should look like "0,1,2" or "0:10" (default: all sequences)'
    )
    parser.add_argument(
        '--output', required=True, default=None,
        help='Output directory for attribution results'
    )
    parser.add_argument(
        '--device', default='cuda',
        help='Device: cuda or cpu (default: cuda)'
    )
    
    args = parser.parse_args()
    
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Setup
        model_path, data_path, pred_path, output_dir = setup_paths(
            args.model, args.data, args.predictions, args.output
        )
        
        # Load
        model, model_config, sequences, labels, usage, metadata, \
            pred_preds, pred_probs, pred_sse, condition_names = load_model_and_data(
            model_path, data_path, pred_path, device=args.device
        )
        
        # Get indices of sequences to compute attributions for
        if args.windows:
            if ':' in args.windows:
                start, end = args.windows.split(':')
                start = int(start) if start else None
                end = int(end) if end else None
                window_indices = list(range(sequences.shape[0]))[start:end]
            else:
                window_indices = args.windows.split(',')
                window_indices = [int(idx) for idx in window_indices]
            print(f"Computing attributions for {len(window_indices)} specified sequences.")
        else:
            window_indices = None
            print("Computing attributions for all sequences.")

        # Compute attributions
        result_splice = compute_attributions(
            model, sequences, labels, metadata, window_indices, pred_preds, device=args.device
        )
        
        # Save attributions
        save_info = save_attributions(result_splice, output_dir)
        
        print(f"Results saved to: {output_dir}")
        print(f"Files created:")
        print(f"  - Donor attributions: donor_attributions.npz")
        print(f"  - Acceptor attributions: acceptor_attributions.npz")
        print(f"  - Complete result dict: result_splice.npz")
        print(f"Summary: {save_info['summary']}")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
    
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
