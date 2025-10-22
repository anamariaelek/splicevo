"""Training script using configuration file."""

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
        usage_files = {k: mmap_dir / f'usage_{k}.mmap' for k in ('alpha', 'beta', 'sse')}

        # Check if memmap files exist, if not create them
        if not (seq_file.exists() and lbl_file.exists() and all(p.exists() for p in usage_files.values())):
            log_fn(f"Memmap files not found in {mmap_dir}")
            create_mmap_datasets(data_path, mmap_dir, log_fn)
        
        # Check again and load memmap files
        if seq_file.exists() and lbl_file.exists() and all(p.exists() for p in usage_files.values()):
            log_fn(f"Loading data with memory mapping from: {mmap_dir}")
            meta_path = mmap_dir / 'metadata.json'
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing metadata.json in {mmap_dir}")
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            sequences = np.memmap(seq_file, dtype=np.float32, mode='r', shape=tuple(meta['sequences_shape']))
            labels = np.memmap(lbl_file, dtype=np.int64, mode='r', shape=tuple(meta['labels_shape']))
            usage_arrays = {
                k: np.memmap(usage_files[k], dtype=np.float32, mode='r', shape=tuple(meta['usage_shape']))
                for k in ('alpha', 'beta', 'sse')
            }
        else:
            raise FileNotFoundError(f"Memmap files not found in {mmap_dir}.")

    if not use_mmap:
        log_fn(f"Loading data from {data_path}...")
        with np.load(data_path) as data:
            sequences = data['sequences']
            labels = data['labels']
            usage_arrays = {
                'alpha': data['usage_alpha'],
                'beta': data['usage_beta'],
                'sse': data['usage_sse'],
            }
    
    log_fn(f"Normalizing usage arrays (method: {normalization_method})...")
    normalized_usage, usage_stats = normalize_usage_arrays(usage_arrays, method=normalization_method)

    log_fn(f"Loaded {len(sequences)} samples")
    log_fn(f"  Sequences shape: {sequences.shape}")
    log_fn(f"  Labels shape: {labels.shape}")
    log_fn(f"  Usage arrays shapes: " +
           ", ".join(f"{k}: {v.shape}" for k, v in normalized_usage.items()))

    return sequences, labels, normalized_usage, usage_stats


def create_datasets(sequences, labels, normalized_usage, config: dict, log_fn=print):
    """Create train/val datasets."""
    split_ratio = config['data'].get('train_split', 0.8)
    n_samples = len(sequences)
    n_train = int(split_ratio * n_samples)
    
    log_fn(f"\nSplitting data: {n_train} train, {n_samples - n_train} val")
    
    train_dataset = SpliceDataset(
        sequences[:n_train],
        labels[:n_train],
        {k: v[:n_train] for k, v in normalized_usage.items()}
    )
    
    val_dataset = SpliceDataset(
        sequences[n_train:],
        labels[n_train:],
        {k: v[n_train:] for k, v in normalized_usage.items()}
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


def create_model(config: dict, n_conditions: int, log_fn=print):
    """Create model from config."""
    model_config = config['model']
    
    # Use config parameters directly as they match SplicevoModel arguments
    model_params = {
        'embed_dim': model_config.get('embed_dim', 128),
        'num_resblocks': model_config.get('num_resblocks', 8),
        'dilation_strategy': model_config.get('dilation_strategy', 'alternating'),
        'alternate': model_config.get('alternate', 2),
        'num_classes': model_config.get('num_classes', 3),
        'n_conditions': n_conditions,
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
    
    return model


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


def save_training_config(config: dict, checkpoint_dir: Path, usage_stats: dict):
    """Save complete training configuration."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    full_config = {
        'timestamp': timestamp,
        'config': config,
        'normalization_stats': usage_stats
    }
    
    config_path = checkpoint_dir / f'training_config_{timestamp}.json'
    with open(config_path, 'w') as f:
        json.dump(full_config, f, indent=2)
    
    return config_path


def main():
    parser = argparse.ArgumentParser(description='Train splice site model with config file')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file (JSON or YAML)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Override checkpoint directory from config')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device from config')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (path to .pt file, or "auto" to find latest)')
    
    args = parser.parse_args()
    
    # Start timing
    script_start_time = time.time()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Override with command line args
    if args.checkpoint_dir:
        config['output']['checkpoint_dir'] = args.checkpoint_dir
    if args.device:
        config['device'] = args.device
    
    # Setup output directory
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = checkpoint_dir / f'training_log_{timestamp}.txt'
    
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
    
    # Save normalization stats
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
    
    # Create model
    model_start = time.time()
    n_conditions = normalized_usage['alpha'].shape[2]
    model = create_model(config, n_conditions, log_print)
    model_time = time.time() - model_start
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, config, device
    )
    
    log_print(f"\nTraining samples: {len(train_dataset)}")
    log_print(f"Validation samples: {len(val_dataset)}")
    log_print(f"Batches per epoch: {len(train_loader)}")
    
    # Save full configuration
    config_path = save_training_config(config, checkpoint_dir, usage_stats)
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
        usage_weight=training_config.get('usage_weight', 0.5),
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
    
    log_print("="*60)
    log_print(f"\nTraining completed!")
    log_print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    log_print(f"Checkpoints saved to: {checkpoint_dir}")
    log_print(f"\nTiming Summary:")
    log_print(f"  Data loading:     {format_time(data_time):>12} ({100*data_time/total_time:5.1f}%)")
    log_print(f"  Dataset creation: {format_time(dataset_time):>12} ({100*dataset_time/total_time:5.1f}%)")
    log_print(f"  Model init:       {format_time(model_time):>12} ({100*model_time/total_time:5.1f}%)")
    log_print(f"  Training:         {format_time(train_time):>12} ({100*train_time/total_time:5.1f}%)")
    log_print(f"  TOTAL TIME:       {format_time(total_time):>12}")
    log_print(f"\nTraining ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
