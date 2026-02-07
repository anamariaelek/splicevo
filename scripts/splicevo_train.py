"""Training script using configuration file."""

from html import parser
from typing import Dict, Optional
import torch
import numpy as np
import pandas as pd
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
from splicevo.training.samplers import SpeciesBatchSampler
from splicevo.training.normalization import (
    normalize_usage_arrays,
    save_normalization_stats
)
from splicevo.data.data_loader import (
    load_memmap_data,
    sparse_labels_to_dense_batch
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


def load_data(data_path: str, log_fn=print):
    """Load data for training."""

    data_path = Path(data_path)

    # Check if required files exist
    seq_file = data_path / 'sequences.mmap'
    labels_file = data_path / 'labels.parquet'
    
    if not seq_file.exists():
        raise FileNotFoundError(f"Sequences file not found: {seq_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Sparse labels file not found: {labels_file}")
    
    # Load using module-level function
    log_fn(f"Loading data from: {data_path}")
    sequences, labels, usage_sparse_df, metadata = load_memmap_data(
        data_path,
        load_labels=True,
        load_usage=True
    )

    log_fn(f"Loaded {len(sequences)} samples")    
    log_fn(f"  Sequences: {sequences.shape} sequences")
    if labels is not None:
        log_fn(f"  Labels: {len(labels)} entries (sparse format)")
    
    # Load metadata for conditions and window_size
    meta_path = data_path / 'metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Get window size from metadata
    window_size = meta.get('window_size')
    if window_size is None:
        raise ValueError("window_size not found in metadata.json")
    log_fn(f"  Window size: {window_size}")
    
    # Extract usage conditions from metadata
    usage_conditions = meta.get('usage_conditions', [])
    n_conditions = len(usage_conditions)
    log_fn(f"  Usage conditions: {n_conditions} conditions")
    
    # Load species mapping
    species_mapping = meta.get('species_mapping', {})
    if species_mapping:
        log_fn(f"  Species mapping: {species_mapping}")
    
    # Load species IDs from metadata CSV
    species_ids = None
    if metadata is not None and 'species_id' in metadata.columns:
        species_ids = metadata['species_id'].values.astype(np.int32)
        log_fn(f"  Species IDs loaded from metadata: {len(species_ids)} entries")
    
    
    # Check usage data availability
    if usage_sparse_df is None or len(usage_sparse_df) == 0:
        log_fn("No usage data loaded, training without usage prediction.")
        n_conditions = 0
    else:
        log_fn(f"  Usage data: {len(usage_sparse_df)} entries (sparse format)")

    return sequences, labels, window_size, usage_sparse_df, n_conditions, species_ids, species_mapping


def create_datasets(sequences, labels, window_size, usage_sparse_df, n_conditions, species_ids, config: dict, log_fn=print):
    """Create train/val datasets."""
    split_ratio = config['data'].get('train_split', 0.8)
    n_samples = len(sequences)
    n_train = int(split_ratio * n_samples)
    
    log_fn(f"\nSplitting data: {n_train} train, {n_samples - n_train} val")
    
    # Create index arrays instead of slicing data (keeps memmap intact)
    train_indices = np.arange(0, n_train)
    val_indices = np.arange(n_train, n_samples)
    
    # Pass full arrays + indices to dataset (SpliceDataset will use indices for access)
    train_dataset = SpliceDataset(
        sequences,
        labels,
        window_size,
        usage_sparse_df,
        n_conditions,
        species_ids,
        indices=train_indices
    )
    
    val_dataset = SpliceDataset(
        sequences,
        labels,
        window_size,
        usage_sparse_df,
        n_conditions,
        species_ids,
        indices=val_indices
    )
    
    return train_dataset, val_dataset


def compute_class_weights(labels, window_size, config: dict, n_train: int, log_fn=print):
    """Compute class weights by sampling (avoid loading full dataset)."""
    if not config['training'].get('use_class_weights', True):
        return None
    
    log_fn("\nComputing class weights from sample...")
    
    # Sample 10% of training data to estimate class distribution (avoids loading everything)
    sample_size = min(50000, n_train // 10)
    sample_indices = np.random.choice(n_train, size=sample_size, replace=False)
    
    # Reconstruct dense labels for sampled indices only
    sampled_labels = sparse_labels_to_dense_batch(
        labels,
        sample_indices,
        window_size
    ).flatten()
    
    unique_classes, class_counts = np.unique(sampled_labels, return_counts=True)
    
    log_fn("Class distribution (from sample):")
    for cls, count in zip(unique_classes, class_counts):
        log_fn(f"  Class {cls}: {count:,} ({100*count/len(sampled_labels):.2f}%)")
    
    # Inverse frequency weights
    total_samples = len(sampled_labels)
    class_weights = total_samples / (len(unique_classes) * class_counts)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.FloatTensor(class_weights)
    
    log_fn(f"Class weights (estimated): {class_weights.tolist()}")
    
    return class_weights


def create_model(config: dict, n_conditions: int, species_mapping: Dict[str, int], log_fn=print):
    """Create model from config and return model."""
    model_config = config['model']
    training_config = config['training']
    
    # Get loss types from training config
    splice_loss_type = training_config.get('splice_loss_type', 'cross_entropy')
    usage_loss_type = training_config.get('usage_loss_type', 'weighted_mse')
    
    n_species = len(species_mapping) if species_mapping else 1  
    
    model_params = {
        'embed_dim': model_config.get('embed_dim', 128),
        'num_resblocks': model_config.get('num_resblocks', 8),
        'dilation_strategy': model_config.get('dilation_strategy', 'alternating'),
        'alternate': model_config.get('alternate', 2),
        'context_len': model_config.get('context_len', 4500),
        'pooling_type': model_config.get('pooling_type', 'attention'),
        'num_heads': model_config.get('num_heads', 8),
        'bottleneck_dim': model_config.get('bottleneck_dim', None),  # None defaults to embed_dim
        'dropout': model_config.get('dropout', 0.5),
        'mult_factor': model_config.get('mult_factor', 1.0),
        'mult_factor_learnable': model_config.get('mult_factor_learnable', False),
        'usage_loss_type': usage_loss_type,
        'num_classes': model_config.get('num_classes', 3),
        'n_species': n_species,
        'n_conditions': n_conditions,
    }
    model = SplicevoModel(**model_params)
    
    n_params = sum(p.numel() for p in model.parameters())
    log_fn(f"\nModel created with {n_params:,} parameters")
    log_fn(f"  n_species: {model_params['n_species']}")
    if n_species > 1:
        log_fn(f"  species_mapping: {species_mapping}")
    log_fn(f"  n_conditions: {model_params['n_conditions']}")
    log_fn(f"  num_classes: {model_params['num_classes']}")
    log_fn(f"  embed_dim: {model_params['embed_dim']}")
    log_fn(f"  bottleneck_dim: {model_params['bottleneck_dim'] or model_params['embed_dim']}")
    log_fn(f"  num_resblocks: {model_params['num_resblocks']}")
    log_fn(f"  dilation_strategy: {model_params['dilation_strategy']}")
    log_fn(f"  pooling_type: {model_params['pooling_type']}")
    if model_params['pooling_type'] == 'attention':
        log_fn(f"  num_heads: {model_params['num_heads']}")
    if model_params['pooling_type'] == 'softmax_pool':
        log_fn(f"  mult_factor: {model_params['mult_factor']}")
        log_fn(f"  mult_factor_learnable: {model_params['mult_factor_learnable']}")
    log_fn(f"  splice_loss_type: {splice_loss_type}")
    log_fn(f"  usage_loss_type: {usage_loss_type}")
    
    # Return model
    return model


def create_dataloaders(train_dataset, val_dataset, config: dict, device: str):
    """Create data loaders with species-specific batching."""
    dataloader_config = config['training']['dataloader']
    
    # Use SpeciesBatchSampler to ensure each batch contains only one species
    train_batch_sampler = SpeciesBatchSampler(
        dataset=train_dataset,
        batch_size=dataloader_config['batch_size'],
        shuffle=True,
        balance_species=config['training'].get('balance_species', False),
        drop_last=False
    )
    
    val_batch_sampler = SpeciesBatchSampler(
        dataset=val_dataset,
        batch_size=dataloader_config['batch_size'],
        shuffle=False,
        balance_species=False,
        drop_last=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=dataloader_config.get('num_workers', 4),
        pin_memory=device == 'cuda',
        persistent_workers=dataloader_config.get('num_workers', 4) > 0,
        prefetch_factor=dataloader_config.get('prefetch_factor', 2)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        num_workers=dataloader_config.get('num_workers', 4),
        pin_memory=device == 'cuda',
        persistent_workers=dataloader_config.get('num_workers', 4) > 0,
        prefetch_factor=dataloader_config.get('prefetch_factor', 2)
    )
    
    return train_loader, val_loader


def save_training_config(config: dict, checkpoint_dir: Path, usage_conditions: Optional[list] = None, usage_condition_mapping: Optional[dict] = None):
    """Save complete training configuration including condition mapping."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    full_config = {
        'timestamp': timestamp,
        'config': config,
        'usage_conditions': usage_conditions,
        'usage_condition_mapping': usage_condition_mapping
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
    
    # Data path
    data_path = config['data']['path']

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
    
    # Print full configuration
    log_print("\n" + "="*60)
    log_print("Configuration:")
    log_print("="*60)
    log_print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    log_print("="*60 + "\n")
    
    # Setup device
    device = setup_device(config)
    log_print(f"Device: {device}")
    
    # Load and normalize data
    data_start = time.time()
    sequences, labels, window_size, usage_sparse_df, n_conditions, species_ids, species_mapping = load_data(
        data_path, log_print
    )
    data_time = time.time() - data_start
    
    # Get condition information from data if available
    usage_conditions = None
    usage_condition_mapping = None
    
    if 'usage_info' in config['data']:
        # If user provided condition info in config
        usage_conditions = config['data']['usage_info']
    else:
        # Try to load from train metadata
        data_dir = Path(config['data']['path'])
        if data_dir.is_dir():
            train_meta_file = data_dir / 'metadata.json'
        else:
            train_meta_file = data_dir.parent / 'train' / 'metadata.json'
        
        if train_meta_file.exists():
            log_print(f"Loading condition info from {train_meta_file}")
            with open(train_meta_file, 'r') as f:
                train_meta = json.load(f)
                usage_conditions = train_meta.get('usage_conditions', None)
                usage_condition_mapping = train_meta.get('usage_condition_mapping', None)
                if usage_conditions:
                    log_print(f"Loaded {len(usage_conditions)} conditions: {usage_conditions}")
                else:
                    log_print("Warning: No usage conditions found in train metadata.")
                if usage_condition_mapping:
                    log_print(f"Loaded condition mapping with {len(usage_condition_mapping)} entries")
                else:
                    log_print("Warning: No condition mapping found in config or metadata.")
        
    
    # Create datasets
    dataset_start = time.time()
    train_dataset, val_dataset = create_datasets(
        sequences, labels, window_size, usage_sparse_df, n_conditions, species_ids, config, log_print
    )
    dataset_time = time.time() - dataset_start
    
    # Compute class weights (using sampling to avoid loading all data)
    split_ratio = config['data'].get('train_split', 0.8)
    n_train = int(split_ratio * len(sequences))
    class_weights = compute_class_weights(labels, window_size, config, n_train, log_print)
    
    # Loosses from training config
    training_config = config['training']

    # Parse splice loss type
    splice_loss_type = training_config.get('splice_loss_type', 'cross_entropy')
    if splice_loss_type not in ['cross_entropy', 'focal']:
        raise ValueError(f"splice_loss_type must be 'cross_entropy' or 'focal', got: {splice_loss_type}")
    
    focal_alpha = None
    focal_gamma = 2.0
    class_weights_for_trainer = None
    
    if splice_loss_type == 'focal':
        log_print("\nConfiguring Focal Loss for splice site classification...")

        # Get alpha from config
        focal_alpha_config = training_config.get('focal_alpha', None)
        if focal_alpha_config == 'inverse' or focal_alpha_config == 'balanced':
            # Calculate inverse frequency alpha from class weights
            if class_weights is not None:
                focal_alpha = class_weights
                log_print(f"  Inverse frequency alpha from class weights: {focal_alpha.tolist()}")
            else:
                focal_alpha = None
                log_print("  No class weights available, using equal alpha weights")
        elif focal_alpha_config is not None:
            # Use provided alpha values
            focal_alpha = torch.tensor(focal_alpha_config, dtype=torch.float32)
            log_print(f"  Using provided alpha: {focal_alpha.tolist()}")
        else:
            focal_alpha = None
            log_print("  Using equal alpha weights (None)")

        # Get gamma from config
        focal_gamma = training_config.get('focal_gamma', 2.0)
        log_print(f"  Gamma: {focal_gamma} (focusing parameter)")

    else:
        # Using cross_entropy loss with class weights
        log_print("\nUsing Cross Entropy Loss for splice site classification")
        if class_weights is not None:
            class_weights_for_trainer = class_weights
            log_print(f"Class weights will be applied: {class_weights_for_trainer.tolist()}")
        
    # Parse usage loss type and configuration
    usage_loss_type = training_config.get('usage_loss_type', 'weighted_mse')
    
    if usage_loss_type == 'weighted_mse':
        weighted_mse_extreme_low = training_config.get('weighted_mse_extreme_low', 0.05)
        weighted_mse_extreme_high = training_config.get('weighted_mse_extreme_high', 0.95)
        weighted_mse_extreme_weight = training_config.get('weighted_mse_extreme_weight', 10.0)
        
        log_print(f"\nUsing weighted MSE loss for usage predictions:")
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
        
        log_print(f"\nUsing hybrid loss for usage predictions:")
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
        log_print("\nUsing standard MSE loss for usage predictions")
        trainer_kwargs = {'usage_loss_type': 'mse'}

    # Create model (n_conditions already determined from data)
    model_start = time.time()
    model = create_model(config, n_conditions, species_mapping, log_print)
    model_time = time.time() - model_start
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, config, device
    )
    
    log_print(f"\nTraining samples: {len(train_dataset)}")
    log_print(f"Validation samples: {len(val_dataset)}")
    log_print(f"Batches per epoch: {len(train_loader)}")
    
    # Save full configuration
    config_path = save_training_config(config, checkpoint_dir, usage_conditions, usage_condition_mapping)
    log_print(f"Saved training configuration to: {config_path}")
    
    # Create trainer
    log_print("\nInitializing trainer...")

    trainer = SpliceTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        splice_weight=training_config.get('splice_weight', 1.0),
        usage_weight=training_config.get('usage_weight', 0.5) if n_conditions > 0 else 0.0,
        use_dynamic_loss_balancing=training_config.get('use_dynamic_loss_balancing', False),
        class_weights=class_weights_for_trainer,
        splice_loss_type=splice_loss_type,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        checkpoint_dir=str(checkpoint_dir),
        use_tensorboard=config['output'].get('use_tensorboard', True),
        use_amp=training_config.get('use_amp', True),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        warmup_steps=training_config.get('warmup_steps', 0),
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