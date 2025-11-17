"""Training script using configuration file."""

from html import parser
from typing import Dict, Optional
import torch
import numpy as np
from pathlib import Path
import argparse
import json
import yaml
import sys
from datetime import datetime
import os
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from splicevo.model import SplicevoModel
from splicevo.training.trainer import SpliceTrainer
from splicevo.training.dataset import SpliceDataset
from splicevo.training.normalization import (
    normalize_usage_arrays,
    save_normalization_stats
)
from torch.utils.data import DataLoader


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def setup_device(config: dict) -> str:
    """Setup compute device with optional resource limits."""
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Set CPU thread limits
    num_threads = config.get('cpu_threads', 8)
    torch.set_num_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    
    # GPU optimizations
    if device == 'cuda':
        torch.backends.cudnn.benchmark = config.get('cudnn_benchmark', True)
        torch.cuda.empty_cache()
        
        if 'gpu_memory_fraction' in config:
            torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    
    return device


def create_mmap_datasets(data_path, out_dir, log_fn=print):
    """Create memory-mapped datasets for large data."""

    data = np.load(data_path)
    
    sequences = data['sequences']
    labels = data['labels']
    usage_arrays = {
        'alpha': data['usage_alpha'],
        'beta': data['usage_beta'],
        'sse': data['usage_sse']
    }
    
    # Create memmap directory
    log_fn("Creating memory-mapped files")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as memmap
    seq_mmap = np.memmap(
        out_dir / 'sequences.mmap',
        dtype=np.float32,
        mode='w+',
        shape=sequences.shape
    )
    seq_mmap[:] = sequences[:]
    seq_mmap.flush()
    
    labels_mmap = np.memmap(
        out_dir / 'labels.mmap',
        dtype=np.int64,
        mode='w+',
        shape=labels.shape
    )
    labels_mmap[:] = labels[:]
    labels_mmap.flush()
    
    for key in ['alpha', 'beta', 'sse']:
        usage_mmap = np.memmap(
            out_dir / f'usage_{key}.mmap',
            dtype=np.float32,
            mode='w+',
            shape=usage_arrays[key].shape
        )
        usage_mmap[:] = usage_arrays[key][:]
        usage_mmap.flush()
    
    # Save metadata
    metadata = {
        'sequences_shape': sequences.shape,
        'labels_shape': labels.shape,
        'usage_shape': usage_arrays['alpha'].shape
    }
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    log_fn(f"Memory-mapped files saved to: {out_dir}")


def load_and_normalize_data(config: dict, log_fn=print):
    """Load data and apply normalization."""

    data_path = Path(config['data']['path'])
    normalization_method = config['data']['normalization_method']
    use_mmap = config['data']['use_mmap']
    
    # Optional: specify which usage arrays to load
    usage_types_to_load = config['data'].get('usage_types', ['alpha', 'beta', 'sse'])  # Default: all

    sequences = labels = None
    usage_arrays = None
    normalized_usage = None
    usage_stats = None

    if use_mmap:
        # Resolve mmap directory from config path
        if data_path.is_dir():
            mmap_dir = data_path
        elif data_path.suffix == '.npz':
            mmap_dir = data_path.parent / data_path.stem
        elif data_path.name == 'metadata.json':
            mmap_dir = data_path.parent
        else:
            mmap_dir = data_path

        seq_file = mmap_dir / 'sequences.mmap'
        lbl_file = mmap_dir / 'labels.mmap'
        
        # Check which usage files exist
        usage_files = {
            'alpha': mmap_dir / 'usage_alpha.mmap',
            'beta': mmap_dir / 'usage_beta.mmap',
            'sse': mmap_dir / 'usage_sse.mmap'
        }
        
        available_usage = {k: v for k, v in usage_files.items() 
                          if v.exists() and k in usage_types_to_load}

        # Check if memmap files exist
        if not (seq_file.exists() and lbl_file.exists()):
            log_fn(f"Memmap files not found in {mmap_dir}")
            create_mmap_datasets(data_path, mmap_dir, log_fn)
        
        # Load memmap files
        if seq_file.exists() and lbl_file.exists():
            log_fn(f"Loading data with memory mapping from: {mmap_dir}")
            meta_path = mmap_dir / 'metadata.json'
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing metadata.json in {mmap_dir}")
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            # Parse dtypes from metadata
            seq_dtype = np.dtype(meta.get('sequences_dtype', 'float32'))
            lbl_dtype = np.dtype(meta.get('labels_dtype', 'int8'))
            usage_dtype = np.dtype(meta.get('usage_dtype', 'float32'))
            
            sequences = np.memmap(seq_file, dtype=seq_dtype, mode='r', shape=tuple(meta['sequences_shape']))
            labels = np.memmap(lbl_file, dtype=lbl_dtype, mode='r', shape=tuple(meta['labels_shape']))
            
            # Load available usage arrays
            if available_usage:
                log_fn(f"Loading usage arrays: {list(available_usage.keys())}")
                usage_arrays = {}
                for key, filepath in available_usage.items():
                    usage_arrays[key] = np.memmap(
                        filepath, 
                        dtype=usage_dtype, 
                        mode='r', 
                        shape=tuple(meta['usage_shape'])
                    )
                    log_fn(f"  Loaded {key}: {usage_arrays[key].shape}")
            else:
                log_fn("No usage array files found or requested. Usage prediction will be disabled.")
                usage_arrays = None
        else:
            raise FileNotFoundError(f"Memmap files not found in {mmap_dir}.")

    else:
        # Non-memmap loading
        log_fn(f"Loading data from {data_path}...")
        with np.load(data_path) as data:
            sequences = data['sequences']
            labels = data['labels']
            
            # Load only requested usage arrays
            usage_types_to_load = config['data'].get('usage_types', ['alpha', 'beta', 'sse'])
            usage_arrays = {}
            for key in usage_types_to_load:
                usage_key = f'usage_{key}'
                if usage_key in data:
                    usage_arrays[key] = data[usage_key]
                    log_fn(f"  Loaded {key} from {usage_key}")
            
            if not usage_arrays:
                usage_arrays = None
    
    # Normalize if usage arrays are present and not all zeros
    if usage_arrays is None:
        log_fn("No usage arrays loaded, training without usage prediction.")
        normalized_usage = None
        usage_stats = None
    elif all(np.all(arr == 0) for arr in usage_arrays.values()):
        log_fn("All usage arrays are zero, training without usage prediction.")
        normalized_usage = None
        usage_stats = None
    else:
        if normalization_method == 'none':
            log_fn("Skipping normalization of usage arrays (method: none)")
            normalized_usage = {k: usage_arrays[k] for k in usage_arrays}
            usage_stats = {key: {'transform': 'identity'} for key in usage_arrays}
            usage_stats['method'] = 'none'
        else:
            log_fn(f"Normalizing usage arrays (method: {normalization_method})...")
            normalized_usage, usage_stats = normalize_usage_arrays(usage_arrays, method=normalization_method)

    log_fn(f"Loaded {len(sequences)} samples")
    log_fn(f"  Sequences shape: {sequences.shape}")
    log_fn(f"  Labels shape: {labels.shape}")
    if normalized_usage is not None:
        log_fn(f"  Usage arrays loaded: {list(normalized_usage.keys())}")
        for k, v in normalized_usage.items():
            log_fn(f"    {k}: {v.shape}")
    else:
        log_fn("  No usage arrays present.")

    return sequences, labels, normalized_usage, usage_stats


def create_datasets(sequences, labels, normalized_usage, config: dict, log_fn=print):
    """Create train/val datasets."""
    split_ratio = config['data'].get('train_split', 0.8)
    n_samples = len(sequences)
    n_train = int(split_ratio * n_samples)
    
    log_fn(f"\nSplitting data: {n_train} train, {n_samples - n_train} val")
    
    if normalized_usage is not None:
        train_usage = {k: v[:n_train] for k, v in normalized_usage.items()}
        val_usage = {k: v[n_train:] for k, v in normalized_usage.items()}
    else:
        train_usage = None
        val_usage = None

    train_dataset = SpliceDataset(
        sequences[:n_train],
        labels[:n_train],
        train_usage
    )
    
    val_dataset = SpliceDataset(
        sequences[n_train:],
        labels[n_train:],
        val_usage
    )
    
    return train_dataset, val_dataset


def compute_class_weights(labels, config: dict, log_fn=print):
    """Compute class weights for imbalanced data."""
    if not config['training'].get('use_class_weights', True):
        return None
    
    log_fn("\nComputing class weights...")
    train_labels_flat = labels.flatten()
    unique_classes, class_counts = np.unique(train_labels_flat, return_counts=True)
    
    log_fn("Class distribution:")
    for cls, count in zip(unique_classes, class_counts):
        log_fn(f"  Class {cls}: {count:,} ({100*count/len(train_labels_flat):.2f}%)")
    
    # Inverse frequency weights
    total_samples = len(train_labels_flat)
    class_weights = total_samples / (len(unique_classes) * class_counts)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.FloatTensor(class_weights)
    
    log_fn(f"Class weights: {class_weights.tolist()}")
    
    return class_weights


def create_model(config: dict, normalized_usage: Optional[Dict], log_fn=print):
    """Create model from config and return model with usage info."""
    model_config = config['model']
    
    # Determine number of usage prediction outputs based on loaded data
    if normalized_usage is not None and normalized_usage:
        # Get number of conditions and types from first available array
        first_key = next(iter(normalized_usage.keys()))
        n_conditions = normalized_usage[first_key].shape[2]
        n_usage_types = len(normalized_usage)  # How many of alpha/beta/sse we have
    else:
        n_conditions = 0
        n_usage_types = 0
    
    model_params = {
        'embed_dim': model_config.get('embed_dim', 128),
        'num_resblocks': model_config.get('num_resblocks', 8),
        'dilation_strategy': model_config.get('dilation_strategy', 'alternating'),
        'alternate': model_config.get('alternate', 2),
        'num_classes': model_config.get('num_classes', 3),
        'n_conditions': n_conditions,
        'n_usage_types': n_usage_types,  # NEW: pass number of usage types
        'context_len': model_config.get('context_len', 4500),
        'dropout': model_config.get('dropout', 0.5)
    }
    model = SplicevoModel(**model_params)
    
    n_params = sum(p.numel() for p in model.parameters())
    log_fn(f"\nModel created with {n_params:,} parameters")
    log_fn(f"  embed_dim: {model_params['embed_dim']}")
    log_fn(f"  num_resblocks: {model_params['num_resblocks']}")
    log_fn(f"  dilation_strategy: {model_params['dilation_strategy']}")
    log_fn(f"  num_classes: {model_params['num_classes']}")
    log_fn(f"  n_conditions: {model_params['n_conditions']}")
    if n_usage_types > 0:
        log_fn(f"  usage types: {n_usage_types} ({list(normalized_usage.keys())})")
    
    # Return model and n_conditions for trainer setup
    return model, n_conditions


def create_dataloaders(train_dataset, val_dataset, config: dict, device: str):
    """Create data loaders."""
    dataloader_config = config['training']['dataloader']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader_config['batch_size'],
        shuffle=True,
        num_workers=dataloader_config.get('num_workers', 4),
        pin_memory=device == 'cuda',
        persistent_workers=dataloader_config.get('num_workers', 4) > 0,
        prefetch_factor=dataloader_config.get('prefetch_factor', 2)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataloader_config['batch_size'],
        shuffle=False,
        num_workers=dataloader_config.get('num_workers', 4),
        pin_memory=device == 'cuda',
        persistent_workers=dataloader_config.get('num_workers', 4) > 0,
        prefetch_factor=dataloader_config.get('prefetch_factor', 2)
    )
    
    return train_loader, val_loader


def save_training_config(config: dict, checkpoint_dir: Path, usage_stats: dict, condition_info: Optional[Dict] = None):
    """Save complete training configuration including condition mapping."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    full_config = {
        'timestamp': timestamp,
        'config': config,
        'normalization_stats': usage_stats,
        'condition_info': condition_info  # NEW: save tissue/timepoint mapping
    }
    
    config_path = checkpoint_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(full_config, f, indent=2)
    
    return config_path


def main():
    parser = argparse.ArgumentParser(description='Train splice site model with config file')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file (JSON or YAML)')
    parser.add_argument('--data', type=str, default=None,
                        help='Override data path from config')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Override checkpoint directory from config')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device from config')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (path to .pt file, or "auto" to find latest)')
    parser.add_argument("--quiet", action='store_true', 
                        help="Suppress console output (log to file only)")
    
    args = parser.parse_args()
    
    # Start timing
    script_start_time = time.time()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line args
    if args.checkpoint_dir:
        config['output']['checkpoint_dir'] = args.checkpoint_dir
    if args.device:
        config['device'] = args.device
    if args.data:
        config['data']['path'] = args.data
    
    # Setup output directory
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = checkpoint_dir / f'training_log_{timestamp}.txt'

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
    
    log_print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Configuration: {args.config}")
    log_print(f"Log file: {log_file}")
    
    # Setup device
    device = setup_device(config)
    log_print(f"Device: {device}")
    
    # Load and normalize data
    data_start = time.time()
    sequences, labels, normalized_usage, usage_stats = load_and_normalize_data(
        config, log_print
    )
    data_time = time.time() - data_start
    
    # Get condition information from data if available
    condition_info = None
    if 'usage_info' in config['data']:
        # If user provided condition info in config
        condition_info = config['data']['usage_info']
    else:
        # Try to load from usage_info file created during data processing
        data_dir = Path(config['data']['path']).parent
        usage_info_file = data_dir / 'usage_info_train.json'
        if usage_info_file.exists():
            log_print(f"Loading condition info from {usage_info_file}")
            with open(usage_info_file, 'r') as f:
                usage_info = json.load(f)
                condition_info = usage_info.get('conditions', None)
                log_print(f"Loaded {len(condition_info)} conditions")
        else:
            log_print("Warning: No condition info found. Predictions won't have tissue/timepoint labels.")
    
    # Save normalization stats if alpha and beta were normalized
    if usage_stats:
        stats_path = checkpoint_dir / 'normalization_stats.json'
        save_normalization_stats(usage_stats, stats_path)
        log_print(f"Saved normalization stats to: {stats_path}")
    
    # Create datasets
    dataset_start = time.time()
    train_dataset, val_dataset = create_datasets(
        sequences, labels, normalized_usage, config, log_print
    )
    dataset_time = time.time() - dataset_start
    
    # Compute class weights
    train_labels = labels[:len(train_dataset)]
    class_weights = compute_class_weights(train_labels, config, log_print)
    
    # Create model (pass normalized_usage to determine n_conditions)
    model_start = time.time()
    model, n_conditions = create_model(config, normalized_usage, log_print)
    model_time = time.time() - model_start
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, config, device
    )
    
    log_print(f"\nTraining samples: {len(train_dataset)}")
    log_print(f"Validation samples: {len(val_dataset)}")
    log_print(f"Batches per epoch: {len(train_loader)}")
    
    # Save full configuration
    config_path = save_training_config(config, checkpoint_dir, usage_stats, condition_info)
    log_print(f"Saved training configuration to: {config_path}")
    
    # Create trainer
    log_print("\nInitializing trainer...")
    training_config = config['training']
    
    trainer = SpliceTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        splice_weight=training_config.get('splice_weight', 1.0),
        usage_weight=training_config.get('usage_weight', 0.5) if n_conditions > 0 else 0.0,
        class_weights=class_weights,
        checkpoint_dir=str(checkpoint_dir),
        use_tensorboard=config['output'].get('use_tensorboard', True),
        use_amp=training_config.get('use_amp', True),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1)
    )
    
    # Train
    log_print("\nStarting training...")
    log_print("="*60)
    
    train_start = time.time()
    trainer.train(
        n_epochs=training_config['n_epochs'],
        verbose=True,
        save_best=True,
        early_stopping_patience=training_config.get('early_stopping_patience', None),
        resume_from=args.resume
    )
    train_time = time.time() - train_start
    
    total_time = time.time() - script_start_time

    log_print("=" * 60)
    log_print(f"\nTiming Summary:")
    log_print("=" * 60)
    log_print(f"  Data loading:     {format_time(data_time):>12} ({100*data_time/total_time:5.1f}%)")
    log_print(f"  Dataset creation: {format_time(dataset_time):>12} ({100*dataset_time/total_time:5.1f}%)")
    log_print(f"  Model init:       {format_time(model_time):>12} ({100*model_time/total_time:5.1f}%)")
    log_print(f"  Training:         {format_time(train_time):>12} ({100*train_time/total_time:5.1f}%)")
    log_print("-" * 60)
    log_print(f"  Total time:       {format_time(total_time):>12}")
    log_print(f"\nTraining ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 60)
    log_print(f"\nTraining completed!")
    log_print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    log_print(f"Checkpoints saved to: {checkpoint_dir}")
    
    # Restore stdout/stderr if quiet mode
    if args.quiet:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file_handle.close()
        print(f"Training complete. Log saved to: {log_file}")


if __name__ == '__main__':
    main()
