import json
from pathlib import Path
import torch
from splicevo.model import SplicevoModel

def load_model_and_config(checkpoint_path: str, device: str = 'cpu', verbose: bool = False):
    """Load model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if verbose:
        print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Load config
    config_path = checkpoint_path.parent / 'training_config.json'
    model_config = {}
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            training_config = json.load(f)
        model_config = training_config['config']['model']
    
    # Infer n_conditions
    usage_predictor_keys = [k for k in state_dict.keys() if 'usage_predictor' in k]
    if usage_predictor_keys:
        species_keys = [k for k in usage_predictor_keys if 'species_' in k and 'weight' in k]
        if species_keys:
            weight_shape = state_dict[species_keys[0]].shape
            n_conditions = weight_shape[0]
            model_config['n_conditions'] = n_conditions
    
    # Set defaults
    model_config.setdefault('embed_dim', 128)
    model_config.setdefault('num_resblocks', 8)
    model_config.setdefault('dilation_strategy', 'alternating')
    model_config.setdefault('num_classes', 3)
    model_config.setdefault('context_len', 4500)
    model_config.setdefault('dropout', 0.0)
    model_config.setdefault('n_conditions', 5)
    model_config.setdefault('usage_loss_type', 'weighted_mse')
    
    model_config_for_init = {k: v for k, v in model_config.items() if k != 'usage_types'}
    model = SplicevoModel(**model_config_for_init)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    if verbose:
        print(f"Model loaded successfully!")
        
    return model, model_config
