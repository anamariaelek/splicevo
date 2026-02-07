"""Prediction script for splice site prediction models."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from typing import Dict, Optional
import sys
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from splicevo.model import SplicevoModel
from splicevo.data.data_loader import load_memmap_data


def load_model_and_config(checkpoint_path: str, device: str = 'cpu', log_fn=print):
    """Load model from checkpoint and infer configuration."""
    checkpoint_path = Path(checkpoint_path)
    log_fn(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Try to load training config from same directory
    config_path = checkpoint_path.parent / 'training_config.json'
    condition_info = None
    usage_loss_type = 'weighted_mse'  # default
    usage_types = ['sse']  # default
    
    if config_path.exists():
        log_fn(f"Loading model config from {config_path}...")
        with open(config_path, 'r') as f:
            training_config = json.load(f)
        model_config = training_config['config']['model']
        
        # Get usage loss type from training config
        if 'training' in training_config['config']:
            usage_loss_type = training_config['config']['training'].get('usage_loss_type', 'weighted_mse')
        
        # Get usage types from data config
        if 'data' in training_config['config']:
            usage_types = training_config['config']['data'].get('usage_types', ['sse', 'alpha', 'beta'])
            usage_types = [utype for utype in usage_types if utype in ['sse', 'alpha', 'beta']]  # validate types
        else:
            log_fn("Warning: No usage_types found in config, using default [sse, alpha, beta]")
        
        # Load condition info if available
        if 'condition_info' in training_config:
            condition_info = training_config['condition_info']
            log_fn(f"Loaded condition mapping: {len(condition_info)} conditions")
            for i, cond in enumerate(condition_info[:3]):  # Show first 3
                log_fn(f"  Condition {i}: {cond}")
            if len(condition_info) > 3:
                log_fn(f"  ... and {len(condition_info) - 3} more")
    else:
        log_fn("Warning: training_config.json not found, inferring from checkpoint...")
        model_config = {}
        
        # Try to infer usage_loss_type from checkpoint
        # If usage_classifier exists, it was trained with hybrid loss
        if 'encoder.usage_classifier.weight' in state_dict:
            usage_loss_type = 'hybrid'

    # Infer n_conditions from checkpoint state_dict
    usage_predictor_keys = [k for k in state_dict.keys() if 'usage_predictor' in k]
    
    if usage_predictor_keys:
        # Look for species-specific usage predictors
        species_keys = [k for k in usage_predictor_keys if 'species_' in k and 'weight' in k]
        
        if species_keys:
            # Get number of species
            n_species = len(species_keys)
            model_config['n_species'] = n_species
            # Get shape from first species predictor
            sample_key = species_keys[0]
            weight_shape = state_dict[sample_key].shape
            n_conditions = weight_shape[0]
            model_config['n_conditions'] = n_conditions
        else:
            log_fn("Warning: Found usage_predictor but no species-specific predictors")
            model_config.setdefault('n_species', 1)
            model_config.setdefault('n_conditions', 0)
    else:
        log_fn("Warning: Could not find usage_predictor in checkpoint")
        model_config.setdefault('n_species', 1)
        model_config.setdefault('n_conditions', 0)
    
    # Set defaults for other model parameters
    #model_config.setdefault('embed_dim', 128)
    #model_config.setdefault('num_resblocks', 8)
    #model_config.setdefault('dilation_strategy', 'alternating')
    #model_config.setdefault('alternate', 2)
    #model_config.setdefault('num_classes', 3)
    #model_config.setdefault('context_len', 4500)
    #model_config.setdefault('dropout', 0.0)
    #model_config['usage_loss_type'] = usage_loss_type
    
    log_fn("\nModel configuration:")
    for key, value in model_config.items():
        log_fn(f"  {key}: {value}")
    
    # Create model with inferred config 
    model_config_for_init = {k: v for k, v in model_config.items()}
    
    from splicevo.model import SplicevoModel
    model = SplicevoModel(**model_config_for_init)
    
    # Load state dict (strict=False to handle optional usage_classifier)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    log_fn(f"Model loaded successfully!")
    
    return model, model_config, condition_info


def load_test_data(data_path: str, log_fn=print):
    """Load test dataset using sparse format (same as training script)."""
    data_path = Path(data_path)
    
    log_fn(f"\nLoading test data from {data_path}...")
    
    # Check if required files exist
    seq_file = data_path / 'sequences.mmap'
    labels_file = data_path / 'labels.parquet'
    
    if not seq_file.exists():
        raise FileNotFoundError(f"Sequences file not found: {seq_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Sparse labels file not found: {labels_file}")
    
    # Load using the same function as training
    sequences, labels_sparse_df, usage_sparse_df, metadata = load_memmap_data(
        data_path,
        load_labels=True,
        load_usage=True
    )
    
    log_fn(f"  Sequences: {sequences.shape}")
    if labels_sparse_df is not None:
        log_fn(f"  Splice labels: {len(labels_sparse_df)} entries (sparse format)")
    if usage_sparse_df is not None:
        log_fn(f"  Usage data: {len(usage_sparse_df)} entries (sparse format)")
    
    # Load metadata
    meta_path = data_path / 'metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    window_size = meta.get('window_size')
    if window_size is None:
        raise ValueError("window_size not found in metadata.json")
    log_fn(f"  Window size: {window_size}")
    
    # Get species mapping if available
    species_mapping = meta.get('species_mapping', {})
    if species_mapping:
        log_fn(f"  Species mapping: {species_mapping}")
    
    # Load species IDs from metadata CSV
    species_ids = None
    if metadata is not None and 'species_id' in metadata.columns:
        species_ids = metadata['species_id'].values.astype(np.int32)
        log_fn(f"  Species IDs: {len(species_ids)} entries (unique ids: {np.unique(species_ids)})")
    else:
        log_fn(f"  No species_id column found in {metadata}, check your file!")
    
    log_fn(f"Loaded {len(sequences)} samples")

    return {
        'sequences': sequences,
        'labels_sparse': labels_sparse_df,
        'usage_sparse': usage_sparse_df,
        'species_ids': species_ids,
        'window_size': window_size,
        'species_mapping': species_mapping,
        'metadata': metadata
    }


def save_predictions_sparse(
    labels_sparse_df: pd.DataFrame,
    probs_sparse_df: pd.DataFrame,
    usage_sparse_df: pd.DataFrame,
    output_dir: Path,
    condition_info: Optional[dict] = None,
    log_fn=print
):
    """Save predictions in sparse format (parquet files)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_fn(f"\nSaving predictions to {output_dir}...")
    
    # Save splice predictions
    labels_path = output_dir / 'labels_pred.parquet'
    labels_sparse_df.to_parquet(labels_path, index=False, compression='snappy')
    log_fn(f"  Splice predictions: {len(labels_sparse_df)} entries -> {labels_path.name}")
    
    # Save splice probabilities
    if len(probs_sparse_df) > 0:
        probs_path = output_dir / 'probs_pred.parquet'
        probs_sparse_df.to_parquet(probs_path, index=False, compression='snappy')
        log_fn(f"  Splice probabilities: {len(probs_sparse_df)} entries -> {probs_path.name}")
    
    # Save usage predictions
    if len(usage_sparse_df) > 0:
        usage_path = output_dir / 'usage_pred.parquet'
        usage_sparse_df.to_parquet(usage_path, index=False, compression='snappy')
        log_fn(f"  Usage predictions: {len(usage_sparse_df)} entries -> {usage_path.name}")
    
    # Save metadata
    metadata = {
        'format': 'sparse',
        'labels_entries': len(labels_sparse_df),
        'probs_entries': len(probs_sparse_df),
        'usage_entries': len(usage_sparse_df),
        'labels_columns': list(labels_sparse_df.columns) if len(labels_sparse_df) > 0 else [],
        'probs_columns': list(probs_sparse_df.columns) if len(probs_sparse_df) > 0 else [],
        'usage_columns': list(usage_sparse_df.columns) if len(usage_sparse_df) > 0 else []
    }
    
    if condition_info is not None:
        metadata['conditions'] = condition_info
    
    with open(output_dir / 'predictions_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log_fn(f"Predictions saved successfully in sparse format!")


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained splice site model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save predictions directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--gpu-memory-fraction', type=float, default=None,
                        help='Fraction of GPU memory to use (0.0-1.0). If not set, uses all available memory.')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress console output (log to file only)')
    
    args = parser.parse_args()
    
    # Start timing
    script_start_time = time.time()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_dir / f'prediction_log_{timestamp}.txt'
    
    # Redirect stdout and stderr to log file if quiet mode
    if args.quiet:
        # Open log file with line buffering
        log_file_handle = open(log_file, 'a', buffering=1)
        # Save original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        # Redirect
        sys.stdout = log_file_handle
        sys.stderr = log_file_handle
        
        def log_print(msg):
            """Only write to file (stdout already redirected)."""
            print(msg, flush=True)
    else:
        def log_print(msg):
            """Print and write to log file."""
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
    
    def format_time(seconds):
        """Format seconds as hours, minutes, seconds."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
    
    try:
        log_print(f"Prediction started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"Checkpoint: {args.checkpoint}")
        log_print(f"Test data: {args.test_data}")
        log_print(f"Output: {args.output}")
        log_print(f"Log file: {log_file}")
        log_print("=" * 60)
        
        # Set GPU memory limit if specified
        device = args.device if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and args.gpu_memory_fraction is not None:
            if 0.0 < args.gpu_memory_fraction <= 1.0:
                torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
                log_print(f"GPU memory limit set to {args.gpu_memory_fraction:.1%} of available memory")
            else:
                log_print(f"Warning: Invalid gpu_memory_fraction {args.gpu_memory_fraction}, must be in (0.0, 1.0]")
        
        # Load model
        load_start = time.time()
        log_print(f"Device: {device}")
        model, model_config, condition_info = load_model_and_config(
            args.checkpoint, 
            device=device, 
            log_fn=log_print
        )
        load_time = time.time() - load_start
        
        # Load test data
        data_start = time.time()
        test_data = load_test_data(args.test_data, log_fn=log_print)
        sequences = test_data['sequences']
        species_ids = test_data.get('species_ids')
        window_size = test_data['window_size']
        n_samples = sequences.shape[0]
        seq_len = sequences.shape[1]
        data_time = time.time() - data_start
        
        # Calculate central region length
        central_len = seq_len - 2 * model_config['context_len']
        n_conditions = model_config.get('n_conditions', 0)
        
        # Make predictions
        log_print(f"\nMaking predictions on {n_samples} samples...")
        log_print(f"  Sequence length: {seq_len}, Central length: {central_len}")
        log_print(f"  Conditions: {n_conditions}")
        log_print(f"  Using batch size: {args.batch_size}")
        log_print(f"  Memory-mapped data: {isinstance(sequences, np.memmap)}")
        
        predict_start = time.time()
        
        # Run predictions in memory (with sparse format)
        log_print("\nRunning predictions...")
        predictions = model.predict(
            sequences,
            species_ids=species_ids,
            batch_size=args.batch_size,
            sparse_format=True
        )
        predict_time = time.time() - predict_start
        
        labels_sparse_pred = predictions['labels_sparse']
        probs_sparse_pred = predictions['probs_sparse']
        usage_sparse_pred = predictions['usage_sparse']
        
        log_print(f"Predictions complete!")
        log_print(f"  Splice predictions: {len(labels_sparse_pred)} non-zero entries")
        log_print(f"  Splice probabilities: {len(probs_sparse_pred)} entries")
        log_print(f"  Usage predictions: {len(usage_sparse_pred)} entries")
        
        # Save predictions
        save_start = time.time()
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions in sparse format
        save_predictions_sparse(
            labels_sparse_pred,
            probs_sparse_pred,
            usage_sparse_pred,
            output_dir,
            condition_info=condition_info,
            log_fn=log_print
        )
        
        save_time = time.time() - save_start
        
        log_print(f"\nPrediction Summary:")
        log_print(f"  Total samples: {n_samples}")
        log_print(f"  Splice predictions: {len(labels_sparse_pred)} non-zero entries")
        log_print(f"  Usage predictions: {len(usage_sparse_pred)} entries")
        log_print(f"  Output directory: {output_dir}")
        
        # Timing summary
        total_time = time.time() - script_start_time
        
        log_print("")
        log_print("=" * 60)
        log_print("Timing Summary")
        log_print("=" * 60)
        log_print(f"  Model loading:    {format_time(load_time):>12} ({100*load_time/total_time:5.1f}%)")
        log_print(f"  Data loading:     {format_time(data_time):>12} ({100*data_time/total_time:5.1f}%)")
        log_print(f"  Prediction:       {format_time(predict_time):>12} ({100*predict_time/total_time:5.1f}%)")
        log_print(f"  Saving results:   {format_time(save_time):>12} ({100*save_time/total_time:5.1f}%)")
        log_print("-" * 60)
        log_print(f"  Total time:       {format_time(total_time):>12}")
        log_print(f"\nPrediction ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_print("=" * 60)
    
        log_print("\nDone!")
        
    except Exception as e:
        log_print(f"\nError occurred: {str(e)}")
        import traceback
        log_print(traceback.format_exc())
        raise
        
    finally:
        # Restore stdout/stderr if quiet mode
        if args.quiet:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file_handle.close()
            print(f"Predictions complete. Log saved to: {log_file}")


if __name__ == '__main__':
    main()