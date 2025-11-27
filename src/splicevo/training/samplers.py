"""Custom samplers for multi-species training."""

import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Iterator, List, Optional
from collections import defaultdict


class SpeciesBatchSampler(Sampler):
    """
    Batch sampler that ensures each batch contains samples from only one species.
    
    This is essential for Borzoi-style multi-species training where different
    species use different output heads. By keeping batches species-pure, we can
    efficiently select the appropriate head for each batch.
    
    The sampler:
    1. Groups all samples by species
    2. For each epoch, creates batches within each species group
    3. Shuffles the order of batches across species
    4. Optionally balances species representation
    
    Example:
        >>> dataset = YourDataset(...)  # Must have 'species_id' in __getitem__
        >>> sampler = SpeciesBatchSampler(
        ...     dataset=dataset,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     balance_species=True
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        balance_species: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize species-specific batch sampler.
        
        Args:
            dataset: Dataset with 'species_id' field
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples within species groups
            balance_species: Whether to balance species representation
                           If True, samples each species equally (by oversampling rare species)
            drop_last: Whether to drop incomplete batches at the end
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.balance_species = balance_species
        self.drop_last = drop_last
        self.seed = seed
        
        # Group indices by species
        self.species_indices = self._group_by_species()
        self.species_ids = list(self.species_indices.keys())
        
        # Calculate samples per species
        self.species_counts = {
            species_id: len(indices)
            for species_id, indices in self.species_indices.items()
        }
        
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
    
    def _group_by_species(self) -> dict:
        """Group dataset indices by species ID."""
        species_indices = defaultdict(list)
        
        # Iterate through dataset to get species IDs
        for idx in range(len(self.dataset)):
            # Get species_id from dataset
            # Assumes dataset returns dict with 'species_id' key
            sample = self.dataset[idx]
            if isinstance(sample, dict):
                species_id = sample.get('species_id', 0)
            else:
                # If dataset returns tuple, assume species_id is last element
                species_id = sample[-1] if len(sample) > 1 else 0
            
            # Convert to int if tensor
            if isinstance(species_id, torch.Tensor):
                species_id = species_id.item()
            
            species_indices[species_id].append(idx)
        
        return dict(species_indices)
    
    def _create_batches_for_species(
        self,
        species_id: int,
        num_samples: Optional[int] = None
    ) -> List[List[int]]:
        """
        Create batches for a single species.
        
        Args:
            species_id: Species identifier
            num_samples: Number of samples to draw (for balancing)
                        If None, uses all available samples
        
        Returns:
            List of batches (each batch is a list of indices)
        """
        indices = self.species_indices[species_id].copy()
        
        # Shuffle if requested
        if self.shuffle:
            self.rng.shuffle(indices)
        
        # Balance by sampling with replacement if needed
        if num_samples is not None and num_samples > len(indices):
            # Oversample to reach target
            additional = num_samples - len(indices)
            extra_indices = self.rng.choice(indices, size=additional, replace=True)
            indices.extend(extra_indices.tolist())
            if self.shuffle:
                self.rng.shuffle(indices)
        elif num_samples is not None:
            # Subsample to reach target
            indices = indices[:num_samples]
        
        # Create batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            
            # Skip incomplete batches if drop_last
            if self.drop_last and len(batch) < self.batch_size:
                continue
            
            batches.append(batch)
        
        return batches
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches."""
        all_batches = []
        
        if self.balance_species:
            # Calculate target samples per species (max count)
            max_count = max(self.species_counts.values())
            
            # Create balanced batches for each species
            for species_id in self.species_ids:
                batches = self._create_batches_for_species(
                    species_id,
                    num_samples=max_count
                )
                all_batches.extend(batches)
        else:
            # Create batches for each species without balancing
            for species_id in self.species_ids:
                batches = self._create_batches_for_species(species_id)
                all_batches.extend(batches)
        
        # Shuffle batch order across species
        if self.shuffle:
            self.rng.shuffle(all_batches)
        
        # Yield batches
        for batch in all_batches:
            yield batch
    
    def __len__(self) -> int:
        """Return number of batches."""
        total_batches = 0
        
        if self.balance_species:
            max_count = max(self.species_counts.values())
            for species_id in self.species_ids:
                n_samples = max_count
                n_batches = n_samples // self.batch_size
                if not self.drop_last and n_samples % self.batch_size > 0:
                    n_batches += 1
                total_batches += n_batches
        else:
            for species_id, indices in self.species_indices.items():
                n_samples = len(indices)
                n_batches = n_samples // self.batch_size
                if not self.drop_last and n_samples % self.batch_size > 0:
                    n_batches += 1
                total_batches += n_batches
        
        return total_batches
    
    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling with different order each epoch."""
        if self.seed is not None:
            self.rng = np.random.RandomState(self.seed + epoch)


class StratifiedSpeciesBatchSampler(SpeciesBatchSampler):
    """
    Batch sampler that cycles through species in a round-robin fashion.
    
    Instead of shuffling all batches together, this sampler ensures species
    are visited in a regular pattern. Useful for more predictable training
    dynamics.
    
    Example:
        Species A: batches [A1, A2, A3]
        Species B: batches [B1, B2]
        Species C: batches [C1, C2, C3, C4]
        
        Output order: [A1, B1, C1, A2, B2, C2, A3, C3, C4]
    """
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches in round-robin species order."""
        # Create batches for each species
        species_batches = {}
        
        if self.balance_species:
            max_count = max(self.species_counts.values())
            for species_id in self.species_ids:
                species_batches[species_id] = self._create_batches_for_species(
                    species_id,
                    num_samples=max_count
                )
        else:
            for species_id in self.species_ids:
                species_batches[species_id] = self._create_batches_for_species(species_id)
        
        # Round-robin through species
        max_batches = max(len(batches) for batches in species_batches.values())
        
        for batch_idx in range(max_batches):
            # Optionally shuffle species order for this round
            species_order = self.species_ids.copy()
            if self.shuffle:
                self.rng.shuffle(species_order)
            
            for species_id in species_order:
                if batch_idx < len(species_batches[species_id]):
                    yield species_batches[species_id][batch_idx]
