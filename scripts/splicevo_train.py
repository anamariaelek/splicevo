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
    species_ids = None
    species_mapping = {}
    usage_conditions = []

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
        species_file = mmap_dir / 'species_ids.mmap'

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
            usage_dtype = np.dtype(meta.get('usage_dtype', meta.get('sse_dtype', 'float32')))
            usage_shape = tuple(meta.get('usage_shape', meta.get('sse_shape', [])))
            
            # Extract usage conditions from metadata
            usage_conditions = meta.get('usage_conditions', [])
            log_fn(f"  Usage conditions from metadata: {len(usage_conditions)} conditions")
            
            sequences = np.memmap(seq_file, dtype=seq_dtype, mode='r', shape=tuple(meta['sequences_shape']))
            labels = np.memmap(lbl_file, dtype=lbl_dtype, mode='r', shape=tuple(meta['labels_shape']))
            
            # Load SSE if it exists
            if sse_file.exists():
                usage_sse = np.memmap(
                    sse_file, 
                    dtype=usage_dtype, 
                    mode='r', 
                    shape=usage_shape
                )
            else:
                log_fn("No SSE array file found. Usage prediction will be disabled.")
                usage_sse = None

            # Load species IDs if available
            species_file = mmap_dir / 'species_ids.mmap'
            if species_file.exists():
                species_ids = np.memmap(
                    species_file,
                    dtype=np.dtype(meta.get('species_ids_dtype', 'int32')),
                    mode='r',
                    shape=tuple(meta.get('species_ids_shape', [meta['sequences_shape'][0]]))
                )
                log_fn(f"  Species IDs shape: {species_ids.shape}")
                
                # Load species mapping
                species_mapping = meta.get('species_mapping', {})
                log_fn(f"  Species mapping: {species_mapping}")
            else:
                species_ids = None
                species_mapping = {}
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
            
            # Load species IDs if available
            if 'species_ids' in data:
                species_ids = data['species_ids']
                log_fn(f"  Species IDs shape: {species_ids.shape}")
            else:
                species_ids = None
        
        # Try to load species mapping and conditions from metadata
        metadata_path = data_path.parent / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                species_mapping = meta.get('species_mapping', {})
                usage_conditions = meta.get('usage_conditions', [])
        else:
            species_mapping = {}

    # Check if SSE is available and log once
    if usage_sse is None:
        log_fn("No SSE array loaded, training without usage prediction.")
        usage_conditions = []  # No conditions if no SSE
    else:
        # SSE is already in [0,1] range, no normalization needed
        log_fn(f"Loaded SSE array:")
        log_fn(f"  shape={usage_sse.shape}, dtype={usage_sse.dtype}")
        log_fn(f"  conditions={usage_conditions}")
        
        # Sample a small subset to check properties (avoid loading entire memmap into RAM!)
        sample_size = min(1000, usage_sse.shape[0])
        sample_indices = np.random.choice(usage_sse.shape[0], size=sample_size, replace=False)
        sse_sample = usage_sse[sample_indices, :100, :].flatten()  # Sample first 100 positions too
        sse_sample_clean = sse_sample[~np.isnan(sse_sample)]
        
        if len(sse_sample_clean) > 0:
            log_fn(f"  sampled range=[{sse_sample_clean.min():.3f}, {sse_sample_clean.max():.3f}], mean={sse_sample_clean.mean():.3f}")
            
            # Check if sample is all zeros (might indicate bad data)
            if np.all(sse_sample_clean == 0):
                log_fn("WARNING: Sampled SSE values are all zeros - usage predictions may not train properly")
        else:
            log_fn("WARNING: All sampled SSE values are NaN")

    log_fn(f"Loaded {len(sequences)} samples")
    log_fn(f"  Sequences shape: {sequences.shape}")
    log_fn(f"  Labels shape: {labels.shape}")
    if usage_sse is not None:
        log_fn(f"  SSE shape: {usage_sse.shape}")
    else:
        log_fn("  No SSE array present.")

    return sequences, labels, usage_sse, None, species_ids, species_mapping


def create_datasets(sequences, labels, usage_sse, species_ids, config: dict, log_fn=print):
    """Create train/val datasets with index-based streaming (no data copying)."""
    split_ratio = config['data'].get('train_split', 0.8)
    n_samples = len(sequences)
    n_train = int(split_ratio * n_samples)
    
    log_fn(f"\nSplitting data: {n_train} train, {n_samples - n_train} val")
    log_fn("  Using index-based streaming (no data copied to RAM)")
    
    # Create index arrays instead of slicing data (keeps memmap intact)
    train_indices = np.arange(0, n_train)
    val_indices = np.arange(n_train, n_samples)
    
    # Pass full arrays + indices to dataset (SpliceDataset will use indices for access)
    train_dataset = SpliceDataset(
        sequences,
        labels,
        usage_sse,
        species_ids,
        indices=train_indices
    )
    
    val_dataset = SpliceDataset(
        sequences,
        labels,
        usage_sse,
        species_ids,
        indices=val_indices
    )
    
    return train_dataset, val_dataset


def compute_class_weights(labels, config: dict, n_train: int, log_fn=print):
    """Compute class weights by sampling (avoid loading full dataset)."""
    if not config['training'].get('use_class_weights', True):
        return None
    
    log_fn("\nComputing class weights from sample...")
    
    # Sample 10% of training data to estimate class distribution (avoids loading everything)
    sample_size = min(50000, n_train // 10)
    sample_indices = np.random.choice(n_train, size=sample_size, replace=False)
    
    # Load only sampled labels
    sampled_labels = labels[sample_indices].flatten()
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


def create_model(config: dict, usage_sse: Optional[np.ndarray], species_mapping: Dict[str, int], log_fn=print):
    """Create model from config and return model with usage info."""
    model_config = config['model']
    training_config = config['training']
    
    # Determine number of conditions based on loaded data
    if usage_sse is not None:
        # Get number of conditions from SSE array
        n_conditions = usage_sse.shape[2]
    else:
        n_conditions = 0
    
    log_fn(f"\nDetermining model configuration:")
    log_fn(f"  Usage SSE present: {usage_sse is not None}")
    if usage_sse is not None:
        log_fn(f"  SSE shape: {usage_sse.shape}")
        log_fn(f"  Inferred n_conditions: {n_conditions}")
    
    # Get usage loss type from training config
    usage_loss_type = training_config.get('usage_loss_type', 'weighted_mse')
    
    n_species = len(species_mapping) if species_mapping else 1  
    
    model_params = {
        'embed_dim': model_config.get('embed_dim', 128),
        'num_resblocks': model_config.get('num_resblocks', 8),
        'dilation_strategy': model_config.get('dilation_strategy', 'alternating'),
        'alternate': model_config.get('alternate', 2),
        'num_classes': model_config.get('num_classes', 3),
        'n_conditions': n_conditions,
        'context_len': model_config.get('context_len', 4500),
        'num_heads': model_config.get('num_heads', 8),
        'bottleneck_dim': model_config.get('bottleneck_dim', None),  # None defaults to embed_dim
        'dropout': model_config.get('dropout', 0.5),
        'usage_loss_type': usage_loss_type,
        'n_species': n_species
    }
    model = SplicevoModel(**model_params)
    
    n_params = sum(p.numel() for p in model.parameters())
    log_fn(f"\nModel created with {n_params:,} parameters")
    log_fn(f"  embed_dim: {model_params['embed_dim']}")
    log_fn(f"  bottleneck_dim: {model_params['bottleneck_dim'] or model_params['embed_dim']}")
    log_fn(f"  num_resblocks: {model_params['num_resblocks']}")
    log_fn(f"  dilation_strategy: {model_params['dilation_strategy']}")
    log_fn(f"  num_heads: {model_params['num_heads']}")
    log_fn(f"  num_classes: {model_params['num_classes']}")
    log_fn(f"  n_conditions: {model_params['n_conditions']}")
    log_fn(f"  usage_loss_type: {usage_loss_type}")
    log_fn(f"  n_species: {model_params['n_species']}")
    if n_species > 1:
        log_fn(f"  species_mapping: {species_mapping}")
    
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
    sequences, labels, usage_sse, usage_stats, species_ids, species_mapping = load_data(
        config, log_print
    )
    data_time = time.time() - data_start
    
    # Get condition information from data if available
    condition_info = None
    if 'usage_info' in config['data']:
        # If user provided condition info in config
        condition_info = config['data']['usage_info']
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
                condition_info = train_meta.get('usage_conditions', None)
                if condition_info:
                    log_print(f"Loaded {len(condition_info)} conditions: {condition_info}")
        else:
            log_print("Warning: No condition info found in train metadata. Predictions won't have tissue/timepoint labels.")
    
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
        sequences, labels, usage_sse, species_ids, config, log_print
    )
    dataset_time = time.time() - dataset_start
    
    # Compute class weights (using sampling to avoid loading all data)
    split_ratio = config['data'].get('train_split', 0.8)
    n_train = int(split_ratio * len(sequences))
    class_weights = compute_class_weights(labels, config, n_train, log_print)
    
    # Parse splice loss type
    splice_loss_type = config['training'].get('splice_loss_type', 'cross_entropy')
    if splice_loss_type not in ['cross_entropy', 'focal']:
        raise ValueError(f"splice_loss_type must be 'cross_entropy' or 'focal', got: {splice_loss_type}")
    
    focal_alpha = None
    focal_gamma = 2.0
    class_weights_for_trainer = None
    
    if splice_loss_type == 'focal':
        log_print("\nConfiguring Focal Loss for splice site classification...")
        focal_gamma = config['training'].get('focal_gamma', 2.0)
        
        # Get alpha from config
        focal_alpha_config = config['training'].get('focal_alpha', None)
        if focal_alpha_config == 'auto' or focal_alpha_config == 'balanced':
            # Auto-calculate alpha from class weights
            if class_weights is not None:
                focal_alpha = class_weights
                log_print(f"  Auto-calculated alpha from class weights: {focal_alpha.tolist()}")
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
    else:
        # Using cross_entropy loss with class weights
        class_weights_for_trainer = class_weights
        
        log_print(f"  Gamma: {focal_gamma} (focusing parameter)")
        log_print("  Note: Class weights will be ignored when using Focal Loss")
    
    # Create model (pass usage_sse to determine n_conditions)
    model_start = time.time()
    model, n_conditions = create_model(config, usage_sse, species_mapping, log_print)
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
        class_weights=class_weights_for_trainer,
        splice_loss_type=splice_loss_type,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        use_masked_loss=training_config.get('use_masked_loss', False),
        context_window=training_config.get('context_window', 100),
        use_hard_negative_mining=training_config.get('use_hard_negative_mining', False),
        negative_ratio=training_config.get('negative_ratio', 3.0),
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