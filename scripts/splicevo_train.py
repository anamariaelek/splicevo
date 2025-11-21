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
    
    # Only load SSE for training (alpha/beta stay in data processing only)
    usage_sse = data['usage_sse']
    
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
    
    # Only save SSE
    usage_mmap = np.memmap(
        out_dir / 'usage_sse.mmap',
        dtype=np.float32,
        mode='w+',
        shape=usage_sse.shape
    )
    usage_mmap[:] = usage_sse[:]
    usage_mmap.flush()
    
    # Save metadata
    metadata = {
        'sequences_shape': sequences.shape,
        'labels_shape': labels.shape,
        'usage_shape': usage_sse.shape
    }
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    log_fn(f"Memory-mapped files saved to: {out_dir}")


def load_data(config: dict, log_fn=print):
    """Load data."""

    data_path = Path(config['data']['path'])
    use_mmap = config['data']['use_mmap']

    sequences = labels = None
    usage_sse = None

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
        sse_file = mmap_dir / 'usage_sse.mmap'

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
            
            # Load SSE if it exists
            if sse_file.exists():
                usage_sse = np.memmap(
                    sse_file, 
                    dtype=usage_dtype, 
                    mode='r', 
                    shape=tuple(meta['usage_shape'])
                )
            else:
                log_fn("No SSE array file found. Usage prediction will be disabled.")
                usage_sse = None
        else:
            raise FileNotFoundError(f"Memmap files not found in {mmap_dir}.")

    else:
        # Non-memmap loading
        log_fn(f"Loading data from {data_path}...")
        with np.load(data_path) as data:
            sequences = data['sequences']
            labels = data['labels']
            
            # Load SSE if available
            if 'usage_sse' in data:
                usage_sse = data['usage_sse']
            else:
                usage_sse = None
    
    # Check if SSE is available and log once
    if usage_sse is None:
        log_fn("No SSE array loaded, training without usage prediction.")
    elif np.all(usage_sse == 0):
        log_fn("SSE array is all zeros, training without usage prediction.")
        usage_sse = None
    else:
        # SSE is already in [0,1] range, no normalization needed
        log_fn(f"Loaded SSE array:")
        log_fn(f"  shape={usage_sse.shape}, dtype={usage_sse.dtype}")
        sse_clean = usage_sse[~np.isnan(usage_sse)]
        if len(sse_clean) > 0:
            log_fn(f"  range=[{sse_clean.min():.3f}, {sse_clean.max():.3f}], mean={sse_clean.mean():.3f}")

    log_fn(f"Loaded {len(sequences)} samples")
    log_fn(f"  Sequences shape: {sequences.shape}")
    log_fn(f"  Labels shape: {labels.shape}")
    if usage_sse is not None:
        log_fn(f"  SSE shape: {usage_sse.shape}")
    else:
        log_fn("  No SSE array present.")

    return sequences, labels, usage_sse, None


def create_datasets(sequences, labels, usage_sse, config: dict, log_fn=print):
    """Create train/val datasets."""
    split_ratio = config['data'].get('train_split', 0.8)
    n_samples = len(sequences)
    n_train = int(split_ratio * n_samples)
    
    log_fn(f"\nSplitting data: {n_train} train, {n_samples - n_train} val")
    
    if usage_sse is not None:
        train_sse = usage_sse[:n_train]
        val_sse = usage_sse[n_train:]
    else:
        train_sse = None
        val_sse = None

    train_dataset = SpliceDataset(
        sequences[:n_train],
        labels[:n_train],
        train_sse
    )
    
    val_dataset = SpliceDataset(
        sequences[n_train:],
        labels[n_train:],
        val_sse
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


def create_model(config: dict, usage_sse: Optional[np.ndarray], log_fn=print):
    """Create model from config and return model with usage info."""
    model_config = config['model']
    training_config = config['training']
    
    # Determine number of conditions based on loaded data
    if usage_sse is not None:
        # Get number of conditions from SSE array
        n_conditions = usage_sse.shape[2]
    else:
        n_conditions = 0
    
    # Get usage loss type from training config
    usage_loss_type = training_config.get('usage_loss_type', 'weighted_mse')
    
    model_params = {
        'embed_dim': model_config.get('embed_dim', 128),
        'num_resblocks': model_config.get('num_resblocks', 8),
        'dilation_strategy': model_config.get('dilation_strategy', 'alternating'),
        'alternate': model_config.get('alternate', 2),
        'num_classes': model_config.get('num_classes', 3),
        'n_conditions': n_conditions,
        'context_len': model_config.get('context_len', 4500),
        'dropout': model_config.get('dropout', 0.5),
        'usage_loss_type': usage_loss_type
    }
    model = SplicevoModel(**model_params)
    
    n_params = sum(p.numel() for p in model.parameters())
    log_fn(f"\nModel created with {n_params:,} parameters")
    log_fn(f"  embed_dim: {model_params['embed_dim']}")
    log_fn(f"  num_resblocks: {model_params['num_resblocks']}")
    log_fn(f"  dilation_strategy: {model_params['dilation_strategy']}")
    log_fn(f"  num_classes: {model_params['num_classes']}")
    log_fn(f"  n_conditions: {model_params['n_conditions']}")
    log_fn(f"  usage_loss_type: {usage_loss_type}")
    if usage_loss_type == 'hybrid':
        log_fn(f"    -> Classification head added for hybrid loss")
    
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
    sequences, labels, usage_sse, usage_stats = load_data(
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
    if usage_stats is not None:
        stats_path = checkpoint_dir / 'normalization_stats.json'
        save_normalization_stats(usage_stats, stats_path)
        log_print(f"Saved normalization stats to: {stats_path}")
    else:
        log_print("No normalization applied (SSE already in [0,1] range)")
    
    # Create datasets
    dataset_start = time.time()
    train_dataset, val_dataset = create_datasets(
        sequences, labels, usage_sse, config, log_print
    )
    dataset_time = time.time() - dataset_start
    
    # Compute class weights
    train_labels = labels[:len(train_dataset)]
    class_weights = compute_class_weights(train_labels, config, log_print)
    
    # Create model (pass usage_sse to determine n_conditions)
    model_start = time.time()
    model, n_conditions = create_model(config, usage_sse, log_print)
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
    
    # Get usage loss configuration
    usage_loss_type = training_config.get('usage_loss_type', 'weighted_mse')
    
    if usage_loss_type == 'weighted_mse':
        weighted_mse_extreme_low = training_config.get('weighted_mse_extreme_low', 0.05)
        weighted_mse_extreme_high = training_config.get('weighted_mse_extreme_high', 0.95)
        weighted_mse_extreme_weight = training_config.get('weighted_mse_extreme_weight', 10.0)
        
        log_print(f"Using weighted MSE loss for usage predictions:")
        log_print(f"  extreme_low: {weighted_mse_extreme_low} (values < this are 'zero')")
        log_print(f"  extreme_high: {weighted_mse_extreme_high} (values > this are 'one')")
        log_print(f"  extreme_weight: {weighted_mse_extreme_weight}x")
        
        trainer_kwargs = {
            'usage_loss_type': 'weighted_mse',
            'weighted_mse_extreme_low': weighted_mse_extreme_low,
            'weighted_mse_extreme_high': weighted_mse_extreme_high,
            'weighted_mse_extreme_weight': weighted_mse_extreme_weight
        }
    elif usage_loss_type == 'hybrid':
        hybrid_extreme_low = training_config.get('hybrid_extreme_low', 0.05)
        hybrid_extreme_high = training_config.get('hybrid_extreme_high', 0.95)
        hybrid_class_weight = training_config.get('hybrid_class_weight', 1.0)
        hybrid_reg_weight = training_config.get('hybrid_reg_weight', 1.0)
        
        log_print(f"Using hybrid loss for usage predictions:")
        log_print(f"  extreme_low: {hybrid_extreme_low} (classify as 'zero')")
        log_print(f"  extreme_high: {hybrid_extreme_high} (classify as 'one')")
        log_print(f"  class_weight: {hybrid_class_weight}")
        log_print(f"  reg_weight: {hybrid_reg_weight}")
        
        trainer_kwargs = {
            'usage_loss_type': 'hybrid',
            'hybrid_extreme_low': hybrid_extreme_low,
            'hybrid_extreme_high': hybrid_extreme_high,
            'hybrid_class_weight': hybrid_class_weight,
            'hybrid_reg_weight': hybrid_reg_weight
        }
    else:  # 'mse'
        log_print("Using standard MSE loss for usage predictions")
        trainer_kwargs = {'usage_loss_type': 'mse'}
    
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
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        **trainer_kwargs
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