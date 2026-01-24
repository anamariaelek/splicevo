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
from splicevo.attributions.compute import compute_attributions_splice, compute_attributions_usage, save_attributions_for_modisco


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
    pred_preds, pred_probs, pred_sse, meta, _, _, condition_mask = load_predictions(
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
    
    return result_splice


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
        '--sequences', required=False,
        help='Indices of sequences to compute attributions for (optional). ' \
        'It should look like "0,1,2" or "0:10" (default: all sequences)'
    )
    parser.add_argument(
        '--window', type=int, default=200,
        help='Window size around splice site for attributions (default: 200)'
    )
    parser.add_argument(
        '--output', required=True, default=None,
        help='Output directory for attribution results'
    )
    parser.add_argument(
        '--skip-splice-attributions', 
        action='store_true', dest='skip_splice_attributions',
        help="Set this flag to skip splice attribution calculation"
    )
    parser.add_argument(
        '--skip-usage-attributions', 
        action='store_true', dest='skip_usage_attributions',
        help="Set this flag to skip usage attribution calculation"
    )
    parser.add_argument(
        '--share-attributions-across-conditions',
        action='store_true', dest='share_attributions_across_conditions',
        help="Optimize usage attribution computation by computing all conditions in a single forward pass. ~N_CONDITIONS speedup"
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
        if args.sequences:
            if ':' in args.sequences:
                start, end = args.sequences.split(':')
                start = int(start) if start else None
                end = int(end) if end else None
                window_indices = list(range(sequences.shape[0]))[start:end]
            else:
                window_indices = args.sequences.split(',')
                window_indices = [int(idx) for idx in window_indices]
            print(f"Computing attributions for {len(window_indices)} specified sequences.")
        else:
            window_indices = None
            print("Computing attributions for all sequences.")
        
        if not args.skip_splice_attributions:
            
            # Compute splice attributions
            result_splice = compute_attributions_splice(
                model, sequences, labels, metadata,
                window_indices=window_indices,
                predictions=pred_preds,
                filter_by_correct=True,
                device=args.device,
                verbose=True
            )
            
            # Save splice attributions
            save_splice = save_attributions_for_modisco(
                result_splice['attributions'], 
                output_path = f"{output_dir}/splice",
                window=args.window,
                verbose=False)
            print(f"Attributions for splice site classification saved to: {output_dir}")

        else:
            print("Skipped calculating splice classification attributions.")
            
        if not args.skip_usage_attributions:

            # Compute usage attributions
            result_usage = compute_attributions_usage(
                model, sequences, labels, usage, metadata,
                window_indices=window_indices,
                predictions=pred_preds,
                filter_by_correct=True,
                condition_names=condition_names,
                share_attributions_across_conditions=args.share_attributions_across_conditions,
                device=args.device,
                verbose=True
            )
            
            # Save usage attributions
            save_usage = save_attributions_for_modisco(
                result_usage['attributions'], 
                output_path = f"{output_dir}/usage",
                window=args.window,
                condition_names=condition_names,
                verbose=False)
            print(f"Attributions for splice site usage saved to: {output_dir}")      
        
        else:
            print("Skipped calculating splice usage attributions.")

        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
    
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
