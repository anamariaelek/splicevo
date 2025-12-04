"""Prediction script for splice site prediction models."""

import torch
import numpy as np
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
from splicevo.training.normalization import denormalize_usage, load_normalization_stats


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
    usage_types = ['sse', 'alpha', 'beta']  # default: all types
    
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
            log_fn(f"Usage types trained: {usage_types}")
            usage_types = [utype for utype in usage_types if utype in ['sse', 'alpha', 'beta']]  # validate types
            log_fn(f"  -> Using types: {usage_types}")
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
            log_fn("Detected usage_classifier in checkpoint -> using hybrid loss mode")
    
    #log_fn("\n" + "="*60)
    #log_fn("State dict keys:")
    #log_fn("="*60)
    #for key in sorted(state_dict.keys()):
    #    log_fn(f"  {key}: {state_dict[key].shape}")
    
    # Infer n_conditions from checkpoint state_dict
    usage_predictor_keys = [k for k in state_dict.keys() if 'usage_predictor' in k]
    
    if usage_predictor_keys:
        log_fn(f"\nFound usage_predictor keys: {usage_predictor_keys}")
        # Look for species-specific usage predictors
        species_keys = [k for k in usage_predictor_keys if 'species_' in k and 'weight' in k]
        
        if species_keys:
            # Get shape from first species predictor
            sample_key = species_keys[0]
            weight_shape = state_dict[sample_key].shape
            n_conditions = weight_shape[0]
            log_fn(f"Inferred n_conditions from {sample_key}: shape={weight_shape} -> n_conditions={n_conditions}")
            model_config['n_conditions'] = n_conditions
        else:
            log_fn("Warning: Found usage_predictor but no species-specific predictors")
            model_config.setdefault('n_conditions', 0)
    else:
        log_fn("Warning: Could not find usage_predictor in checkpoint")
        model_config.setdefault('n_conditions', 0)
    
    # Set defaults for other model parameters
    model_config.setdefault('embed_dim', 128)
    model_config.setdefault('num_resblocks', 8)
    model_config.setdefault('dilation_strategy', 'alternating')
    model_config.setdefault('alternate', 2)
    model_config.setdefault('num_classes', 3)
    model_config.setdefault('context_len', 4500)
    model_config.setdefault('dropout', 0.0)
    model_config['usage_loss_type'] = usage_loss_type
    
    log_fn("\nModel configuration:")
    for key, value in model_config.items():
        log_fn(f"  {key}: {value}")
    
    # Create model with inferred config (remove usage_types as it's not a model parameter)
    model_config_for_init = {k: v for k, v in model_config.items() if k != 'usage_types'}
    
    from splicevo.model import SplicevoModel
    model = SplicevoModel(**model_config_for_init)
    
    # Load state dict (strict=False to handle optional usage_classifier)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    log_fn(f"Model loaded successfully!")
    log_fn(f"Usage types: {usage_types}")
    
    return model, model_config, condition_info


def load_test_data(data_path: str, use_memmap: bool = False, log_fn=print) -> Dict[str, np.ndarray]:
    """Load test dataset from .npz file or memmap directory."""
    data_path = Path(data_path)
    
    if use_memmap or data_path.is_dir():
        # Load from memmap directory
        if data_path.is_file() and data_path.suffix == '.npz':
            # Convert .npz path to memmap directory
            mmap_dir = data_path.parent / data_path.stem
        else:
            mmap_dir = data_path
        
        log_fn(f"\nLoading test data with memory mapping from {mmap_dir}...")
        
        # Load metadata
        meta_path = mmap_dir / 'metadata.json'
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata.json in {mmap_dir}")
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Load memmap arrays
        seq_dtype = np.dtype(meta.get('sequences_dtype', 'float32'))
        lbl_dtype = np.dtype(meta.get('labels_dtype', 'int8'))
        usage_dtype = np.dtype(meta.get('alpha_dtype', 'float32'))
        usage_shape = tuple(meta['alpha_shape'])

        sequences = np.memmap(
            mmap_dir / 'sequences.mmap',
            dtype=seq_dtype,
            mode='r',
            shape=tuple(meta['sequences_shape'])
        )
        
        labels = np.memmap(
            mmap_dir / 'labels.mmap',
            dtype=lbl_dtype,
            mode='r',
            shape=tuple(meta['labels_shape'])
        )
        
        # Load usage arrays if they exist
        data = {
            'sequences': sequences,
            'labels': labels
        }
        
        usage_files = {
            'usage_alpha': mmap_dir / 'usage_alpha.mmap',
            'usage_beta': mmap_dir / 'usage_beta.mmap',
            'usage_sse': mmap_dir / 'usage_sse.mmap'
        }
        
        for key, filepath in usage_files.items():
            if filepath.exists():
                data[key] = np.memmap(
                    filepath,
                    dtype=usage_dtype,
                    mode='r',
                    shape=usage_shape
                )
        
        log_fn(f"Available keys: {list(data.keys())}")
        log_fn(f"Sequences shape: {sequences.shape}")
        log_fn(f"Splice labels shape: {labels.shape}")
        
        for key in ['usage_alpha', 'usage_beta', 'usage_sse']:
            if key in data:
                log_fn(f"{key.capitalize()} shape: {data[key].shape}")
        
    else:
        # Load from .npz file (in-memory)
        log_fn(f"\nLoading test data from {data_path}...")
        data_npz = np.load(data_path)
        
        data = {key: data_npz[key] for key in data_npz.keys()}
        
        log_fn(f"Available keys: {list(data.keys())}")
        log_fn(f"Sequences shape: {data['sequences'].shape}")
        log_fn(f"Splice labels shape: {data['labels'].shape}")
        
        if 'usage_alpha' in data:
            log_fn(f"Usage alpha shape: {data['usage_alpha'].shape}")
            log_fn(f"Usage beta shape: {data['usage_beta'].shape}")
            log_fn(f"Usage SSE shape: {data['usage_sse'].shape}")
    
    return data


def denormalize_predictions(
    predictions: Dict[str, np.ndarray],
    normalization_stats_path: str,
    n_samples: int
) -> Dict[str, np.ndarray]:
    """
    Denormalize usage predictions back to original scale.
    
    Args:
        predictions: Dictionary with model predictions
        normalization_stats_path: Path to normalization stats JSON
        n_samples: Number of samples (for per-sample denormalization)
    
    Returns:
        Dictionary with denormalized predictions
    """
    print(f"\nDenormalizing predictions using stats from {normalization_stats_path}...")
    
    stats = load_normalization_stats(normalization_stats_path)
    
    usage_preds = predictions['usage_predictions']  # (n_samples, seq_len, n_conditions, 3)
    
    # Denormalize each component (alpha, beta, sse)
    denorm_alpha = np.zeros_like(usage_preds[..., 0])
    denorm_beta = np.zeros_like(usage_preds[..., 1])
    denorm_sse = np.zeros_like(usage_preds[..., 2])
    
    for i in range(n_samples):
        # Alpha (acceptor usage)
        denorm_alpha[i] = denormalize_usage(
            usage_preds[i, :, :, 0],
            stats,
            key='alpha',
            sample_idx=i if stats['method'] == 'per_sample_cpm' else None
        )
        
        # Beta (donor usage)
        denorm_beta[i] = denormalize_usage(
            usage_preds[i, :, :, 1],
            stats,
            key='beta',
            sample_idx=i if stats['method'] == 'per_sample_cpm' else None
        )
        
        # SSE (usually identity transform, but apply anyway)
        denorm_sse[i] = denormalize_usage(
            usage_preds[i, :, :, 2],
            stats,
            key='sse',
            sample_idx=i if stats['method'] == 'per_sample_cpm' else None
        )
        
        if (i + 1) % 100 == 0:
            print(f"  Denormalized {i+1}/{n_samples} samples...")
    
    denormalized = {
        'usage_alpha_original': denorm_alpha,
        'usage_beta_original': denorm_beta,
        'usage_sse_original': denorm_sse
    }
    
    print(f"Denormalization complete!")
    print(f"  Alpha range: [{denorm_alpha.min():.2f}, {denorm_alpha.max():.2f}]")
    print(f"  Beta range: [{denorm_beta.min():.2f}, {denorm_beta.max():.2f}]")
    print(f"  SSE range: [{denorm_sse.min():.2f}, {denorm_sse.max():.2f}]")
    
    return denormalized


def save_predictions_memmap(
    predictions: Dict[str, np.ndarray],
    output_dir: Path,
    save_logits: bool = False,
    condition_info: Optional[list] = None,
    log_fn=print
):
    """Save predictions as memory-mapped files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_fn(f"\nSaving predictions as memmap to {output_dir}...")
    
    # Save each array as memmap
    for key, array in predictions.items():
        if key == 'splice_logits' and not save_logits:
            continue
        
        filepath = output_dir / f'{key}.mmap'
        log_fn(f"  Writing {key}: {array.shape} -> {filepath.name}")
        
        # Create memmap file
        mmap_array = np.memmap(
            filepath,
            dtype=array.dtype,
            mode='w+',
            shape=array.shape
        )
        
        # Write data
        mmap_array[:] = array[:]
        mmap_array.flush()
        del mmap_array
    
    # Save metadata including condition info
    metadata = {
        key: {
            'shape': list(array.shape),
            'dtype': str(array.dtype)
        }
        for key, array in predictions.items()
        if key != 'splice_logits' or save_logits
    }
    
    # Add condition mapping
    if condition_info is not None:
        metadata['conditions'] = condition_info
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log_fn(f"Memmap files saved successfully!")


def save_predictions_npz(
    output_data: Dict[str, np.ndarray],
    output_path: Path,
    condition_info: Optional[list] = None,
    log_fn=print
):
    """Save predictions as compressed .npz file."""
    log_fn(f"\nSaving predictions to {output_path}...")
    np.savez_compressed(output_path, **output_data)
    
    # Also save condition info as separate JSON
    if condition_info is not None:
        condition_file = output_path.with_suffix('.metadata.json')
        with open(condition_file, 'w') as f:
            json.dump({'conditions': condition_info}, f, indent=2)
        log_fn(f"Condition mapping saved to {condition_file}")
    
    log_fn(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained splice site model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test data (.npz file or memmap directory)')
    parser.add_argument('--normalization-stats', type=str, required=False,
                        help='Path to normalization stats (.json file) - not needed for SSE-only models')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save predictions (.npz file or directory for memmap)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--use-memmap', action='store_true',
                        help='Use memory-mapped data loading')
    parser.add_argument('--save-memmap', action='store_true',
                        help='Save predictions as memmap (faster for large datasets)')
    parser.add_argument('--save-logits', action='store_true',
                        help='Save raw logits (large file)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress console output (log to file only)')
    
    args = parser.parse_args()
    
    # Start timing
    script_start_time = time.time()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    if args.save_memmap:
        output_dir = Path(args.output)
        if output_dir.suffix == '.npz':
            output_dir = output_dir.parent / output_dir.stem
        log_dir = output_dir
    else:
        log_dir = output_path.parent
    
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'prediction_log_{timestamp}.txt'
    
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
        
        # Load model
        load_start = time.time()
        device = args.device if torch.cuda.is_available() else 'cpu'
        log_print(f"Device: {device}")
        model, model_config, condition_info = load_model_and_config(
            args.checkpoint, 
            device=device, 
            log_fn=log_print
        )
        load_time = time.time() - load_start
        
        # Load test data
        data_start = time.time()
        test_data = load_test_data(args.test_data, use_memmap=args.use_memmap, log_fn=log_print)
        sequences = test_data['sequences']
        n_samples = sequences.shape[0]
        data_time = time.time() - data_start
        
        # Make predictions
        log_print(f"\nMaking predictions on {n_samples} samples...")
        log_print(f"Using batch size: {args.batch_size}")
        log_print(f"Memory-mapped data: {isinstance(sequences, np.memmap)}")
        
        predict_start = time.time()
        predictions = model.predict(
            sequences,
            batch_size=args.batch_size
        )
        predict_time = time.time() - predict_start
        
        log_print(f"Predictions complete!")
        log_print(f"  Splice logits shape: {predictions['splice_logits'].shape}")
        log_print(f"  Splice probs shape: {predictions['splice_probs'].shape}")
        log_print(f"  Splice predictions shape: {predictions['splice_predictions'].shape}")
        log_print(f"  Usage predictions shape: {predictions['usage_predictions'].shape}")

        # Extract SSE predictions
        usage_preds = predictions['usage_predictions']  # (n_samples, central_len, n_conditions)
        
        log_print(f"\nExtracted usage predictions:")
        log_print(f"  SSE shape: {usage_preds.shape} (sigmoid, range [0,1])")
        
        output_predictions = {
            'splice_predictions': predictions['splice_predictions'],
            'splice_probs': predictions['splice_probs'],
            'usage_sse': usage_preds
        }
        
        if args.save_logits:
            output_predictions['splice_logits'] = predictions['splice_logits']

        # Add ground truth if available
        ground_truth = {}
        if 'labels' in test_data:
            ground_truth['labels_true'] = np.array(test_data['labels'])
        if 'usage_sse' in test_data:
            ground_truth['usage_sse_true'] = np.array(test_data['usage_sse'])
        
        # Save predictions
        save_start = time.time()
        if args.save_memmap:
            # Save as memmap files
            output_dir = Path(args.output)
            if output_dir.suffix == '.npz':
                output_dir = output_dir.parent / output_dir.stem
            
            all_outputs = {**output_predictions, **ground_truth}
            save_predictions_memmap(
                all_outputs, 
                output_dir, 
                args.save_logits, 
                condition_info=condition_info,
                log_fn=log_print
            )
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in output_dir.glob('*.mmap'))
            log_print(f"\nPrediction Summary:")
            log_print(f"  Total samples: {n_samples}")
            log_print(f"  Usage type: SSE only")
            log_print(f"  Output directory: {output_dir}")
            log_print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
            
        else:
            # Save as compressed .npz
            output_path = Path(args.output)
            output_data = {**output_predictions, **ground_truth}
            save_predictions_npz(
                output_data, 
                output_path, 
                condition_info=condition_info,
                log_fn=log_print
            )
            
            log_print(f"\nPrediction Summary:")
            log_print(f"  Total samples: {n_samples}")
            log_print(f"  Usage type: SSE only")
            log_print(f"  Output file: {args.output}")
        
        save_time = time.time() - save_start
        total_time = time.time() - script_start_time
        
        # Timing summary
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