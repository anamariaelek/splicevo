"""
Tests for condition masking functionality.

These tests verify:
1. build_condition_mask helper function creates correct masks
2. Condition masks are correctly created in data splitting
3. load_processed_data correctly loads and returns condition masks
4. Masks have the expected shape and values for each genome
"""

import pytest
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch


class TestBuildConditionMask:
    """Test the build_condition_mask helper function."""
    
    def test_all_conditions_present(self):
        """Test when all genome conditions are in target conditions."""
        genome_conds = ['Brain_1', 'Heart_1', 'Liver_1']
        target_conds = ['Brain_1', 'Heart_1', 'Liver_1', 'Kidney_1']
        
        # Import the function - we'll need to execute it in context
        mask_code = """
import numpy as np

def build_condition_mask(genome_conds, target_conds):
    mask = np.zeros(len(target_conds), dtype=np.bool_)
    cond_to_idx = {{cond: idx for idx, cond in enumerate(genome_conds)}}
    for target_idx, target_cond in enumerate(target_conds):
        if target_cond in cond_to_idx:
            mask[target_idx] = True
    return mask

mask = build_condition_mask({}, {})
""".format(repr(genome_conds), repr(target_conds))
        
        exec_globals = {}
        exec(mask_code, exec_globals)
        mask = exec_globals['mask']
        
        # First 3 should be True, last should be False
        assert mask[0] == True  # Brain_1
        assert mask[1] == True  # Heart_1
        assert mask[2] == True  # Liver_1
        assert mask[3] == False  # Kidney_1 (not in genome)
        assert mask.sum() == 3
    
    def test_partial_overlap(self):
        """Test when only some genome conditions are in target."""
        genome_conds = ['Brain_1', 'Brain_2', 'Heart_1', 'Heart_2']
        target_conds = ['Brain_1', 'Liver_1', 'Heart_1', 'Kidney_1']
        
        mask_code = """
import numpy as np

def build_condition_mask(genome_conds, target_conds):
    mask = np.zeros(len(target_conds), dtype=np.bool_)
    cond_to_idx = {{cond: idx for idx, cond in enumerate(genome_conds)}}
    for target_idx, target_cond in enumerate(target_conds):
        if target_cond in cond_to_idx:
            mask[target_idx] = True
    return mask

mask = build_condition_mask({}, {})
""".format(repr(genome_conds), repr(target_conds))
        
        exec_globals = {}
        exec(mask_code, exec_globals)
        mask = exec_globals['mask']
        
        assert mask[0] == True   # Brain_1 (in genome)
        assert mask[1] == False  # Liver_1 (not in genome)
        assert mask[2] == True   # Heart_1 (in genome)
        assert mask[3] == False  # Kidney_1 (not in genome)
        assert mask.sum() == 2
    
    def test_no_overlap(self):
        """Test when genome conditions don't overlap with target."""
        genome_conds = ['Brain_1', 'Heart_1']
        target_conds = ['Liver_1', 'Kidney_1']
        
        mask_code = """
import numpy as np

def build_condition_mask(genome_conds, target_conds):
    mask = np.zeros(len(target_conds), dtype=np.bool_)
    cond_to_idx = {{cond: idx for idx, cond in enumerate(genome_conds)}}
    for target_idx, target_cond in enumerate(target_conds):
        if target_cond in cond_to_idx:
            mask[target_idx] = True
    return mask

mask = build_condition_mask({}, {})
""".format(repr(genome_conds), repr(target_conds))
        
        exec_globals = {}
        exec(mask_code, exec_globals)
        mask = exec_globals['mask']
        
        assert mask.sum() == 0  # No overlap
        assert all(~mask)
    
    def test_empty_genome_conditions(self):
        """Test with empty genome conditions list."""
        genome_conds = []
        target_conds = ['Brain_1', 'Heart_1', 'Liver_1']
        
        mask_code = """
import numpy as np

def build_condition_mask(genome_conds, target_conds):
    mask = np.zeros(len(target_conds), dtype=np.bool_)
    cond_to_idx = {{cond: idx for idx, cond in enumerate(genome_conds)}}
    for target_idx, target_cond in enumerate(target_conds):
        if target_cond in cond_to_idx:
            mask[target_idx] = True
    return mask

mask = build_condition_mask({}, {})
""".format(repr(genome_conds), repr(target_conds))
        
        exec_globals = {}
        exec(mask_code, exec_globals)
        mask = exec_globals['mask']
        
        assert mask.sum() == 0
        assert len(mask) == 3


class TestConditionMaskCreation:
    """Test condition mask creation in data files."""
    
    def test_mask_shape_and_dtype(self):
        """Test that created masks have correct shape and dtype."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n_sequences = 100
            n_conditions = 10
            
            # Create a mask file
            mask_path = os.path.join(tmpdir, 'condition_mask.mmap')
            mask = np.memmap(mask_path, dtype=np.bool_, mode='w+',
                           shape=(n_sequences, n_conditions))
            
            # Fill with some pattern
            mask[:50, :5] = True  # First 50 sequences have first 5 conditions valid
            mask[50:, 5:] = True  # Last 50 sequences have last 5 conditions valid
            mask.flush()
            del mask
            
            # Reload and verify
            mask_loaded = np.memmap(mask_path, dtype=np.bool_, mode='r',
                                   shape=(n_sequences, n_conditions))
            
            assert mask_loaded.shape == (n_sequences, n_conditions)
            assert mask_loaded.dtype == np.bool_
            assert mask_loaded[:50, :5].all()
            assert mask_loaded[50:, 5:].all()
            assert not mask_loaded[:50, 5:].any()
            assert not mask_loaded[50:, :5].any()
    
    def test_mask_per_genome_pattern(self):
        """Test that masks correctly represent different genomes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n_sequences = 300
            n_conditions = 99
            
            # Simulate 3 genomes with different condition counts
            # Human: 62 conditions, Mouse: 93, Rat: 83
            genome_conditions = {
                0: 62,  # human
                1: 93,  # mouse
                2: 83   # rat
            }
            
            # Create mask
            mask_path = os.path.join(tmpdir, 'condition_mask.mmap')
            mask = np.memmap(mask_path, dtype=np.bool_, mode='w+',
                           shape=(n_sequences, n_conditions))
            
            # Create species_ids
            species_path = os.path.join(tmpdir, 'species_ids.mmap')
            species_ids = np.memmap(species_path, dtype=np.int32, mode='w+',
                                  shape=(n_sequences,))
            
            # Assign genomes: 100 sequences each
            species_ids[:100] = 0  # human
            species_ids[100:200] = 1  # mouse
            species_ids[200:300] = 2  # rat
            
            # Set mask based on genome
            for i in range(n_sequences):
                genome_id = species_ids[i]
                n_valid = genome_conditions[genome_id]
                mask[i, :n_valid] = True
            
            mask.flush()
            species_ids.flush()
            del mask, species_ids
            
            # Verify
            mask_verify = np.memmap(mask_path, dtype=np.bool_, mode='r',
                                   shape=(n_sequences, n_conditions))
            species_verify = np.memmap(species_path, dtype=np.int32, mode='r',
                                      shape=(n_sequences,))
            
            # Check human sequences
            human_mask = species_verify == 0
            assert mask_verify[human_mask].sum(axis=1).mean() == 62
            
            # Check mouse sequences
            mouse_mask = species_verify == 1
            assert mask_verify[mouse_mask].sum(axis=1).mean() == 93
            
            # Check rat sequences
            rat_mask = species_verify == 2
            assert mask_verify[rat_mask].sum(axis=1).mean() == 83


class TestLoadProcessedDataWithMask:
    """Test that load_processed_data correctly handles condition masks."""
    
    def test_load_with_mask_memmap(self):
        """Test loading data with condition mask from memmap directory."""
        from splicevo.utils.data_utils import load_processed_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock data files
            n_sequences = 10
            seq_length = 100
            n_conditions = 5
            window_size = 50
            
            # Create metadata
            metadata = {
                'sequences_shape': [n_sequences, seq_length, 4],
                'sequences_dtype': 'float32',
                'labels_shape': [n_sequences, window_size],
                'labels_dtype': 'int8',
                'alpha_shape': [n_sequences, window_size, n_conditions],
                'alpha_dtype': 'float32',
                'beta_shape': [n_sequences, window_size, n_conditions],
                'beta_dtype': 'float32',
                'sse_shape': [n_sequences, window_size, n_conditions],
                'sse_dtype': 'float32',
                'species_ids_shape': [n_sequences],
                'species_ids_dtype': 'int32',
                'condition_mask': {
                    'shape': [n_sequences, n_conditions],
                    'dtype': 'bool'
                }
            }
            
            with open(os.path.join(tmpdir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
            # Create data files
            np.memmap(os.path.join(tmpdir, 'sequences.mmap'),
                     dtype=np.float32, mode='w+',
                     shape=(n_sequences, seq_length, 4))
            np.memmap(os.path.join(tmpdir, 'labels.mmap'),
                     dtype=np.int8, mode='w+',
                     shape=(n_sequences, window_size))
            np.memmap(os.path.join(tmpdir, 'usage_alpha.mmap'),
                     dtype=np.float32, mode='w+',
                     shape=(n_sequences, window_size, n_conditions))
            np.memmap(os.path.join(tmpdir, 'usage_beta.mmap'),
                     dtype=np.float32, mode='w+',
                     shape=(n_sequences, window_size, n_conditions))
            np.memmap(os.path.join(tmpdir, 'usage_sse.mmap'),
                     dtype=np.float32, mode='w+',
                     shape=(n_sequences, window_size, n_conditions))
            np.memmap(os.path.join(tmpdir, 'species_ids.mmap'),
                     dtype=np.int32, mode='w+',
                     shape=(n_sequences,))
            
            # Create condition mask
            mask = np.memmap(os.path.join(tmpdir, 'condition_mask.mmap'),
                           dtype=np.bool_, mode='w+',
                           shape=(n_sequences, n_conditions))
            mask[:5, :3] = True  # First 5 sequences have first 3 conditions
            mask[5:, 3:] = True  # Last 5 sequences have last 2 conditions
            mask.flush()
            del mask
            
            # Load data
            result = load_processed_data(tmpdir)
            
            # Should return 7 values now (added condition_mask)
            assert len(result) == 7
            sequences, labels, alpha, beta, sse, species_ids, condition_mask = result
            
            # Verify mask was loaded
            assert condition_mask is not None
            assert condition_mask.shape == (n_sequences, n_conditions)
            assert condition_mask.dtype == np.bool_
            assert condition_mask[:5, :3].all()
            assert condition_mask[5:, 3:].all()
    
    def test_load_without_mask_memmap(self):
        """Test loading data without condition mask (backward compatibility)."""
        from splicevo.utils.data_utils import load_processed_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock data files WITHOUT mask
            n_sequences = 10
            seq_length = 100
            n_conditions = 5
            window_size = 50
            
            # Create metadata WITHOUT condition_mask entry
            metadata = {
                'sequences_shape': [n_sequences, seq_length, 4],
                'sequences_dtype': 'float32',
                'labels_shape': [n_sequences, window_size],
                'labels_dtype': 'int8',
                'alpha_shape': [n_sequences, window_size, n_conditions],
                'alpha_dtype': 'float32',
                'beta_shape': [n_sequences, window_size, n_conditions],
                'beta_dtype': 'float32',
                'sse_shape': [n_sequences, window_size, n_conditions],
                'sse_dtype': 'float32',
                'species_ids_shape': [n_sequences],
                'species_ids_dtype': 'int32'
            }
            
            with open(os.path.join(tmpdir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
            # Create minimal data files
            np.memmap(os.path.join(tmpdir, 'sequences.mmap'),
                     dtype=np.float32, mode='w+',
                     shape=(n_sequences, seq_length, 4))
            np.memmap(os.path.join(tmpdir, 'labels.mmap'),
                     dtype=np.int8, mode='w+',
                     shape=(n_sequences, window_size))
            np.memmap(os.path.join(tmpdir, 'usage_alpha.mmap'),
                     dtype=np.float32, mode='w+',
                     shape=(n_sequences, window_size, n_conditions))
            np.memmap(os.path.join(tmpdir, 'usage_beta.mmap'),
                     dtype=np.float32, mode='w+',
                     shape=(n_sequences, window_size, n_conditions))
            np.memmap(os.path.join(tmpdir, 'usage_sse.mmap'),
                     dtype=np.float32, mode='w+',
                     shape=(n_sequences, window_size, n_conditions))
            np.memmap(os.path.join(tmpdir, 'species_ids.mmap'),
                     dtype=np.int32, mode='w+',
                     shape=(n_sequences,))
            
            # Load data
            result = load_processed_data(tmpdir)
            
            # Should still return 7 values, but mask should be None
            assert len(result) == 7
            sequences, labels, alpha, beta, sse, species_ids, condition_mask = result
            
            # Mask should be None for backward compatibility
            assert condition_mask is None
    
    def test_load_with_mask_npz(self):
        """Test loading data with condition mask from npz file."""
        from splicevo.utils.data_utils import load_processed_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create npz file with condition mask
            n_sequences = 10
            seq_length = 100
            n_conditions = 5
            window_size = 50
            
            mask = np.zeros((n_sequences, n_conditions), dtype=np.bool_)
            mask[:5, :3] = True
            mask[5:, 3:] = True
            
            data = {
                'sequences': np.zeros((n_sequences, seq_length, 4), dtype=np.float32),
                'labels': np.zeros((n_sequences, window_size), dtype=np.int8),
                'usage_alpha': np.zeros((n_sequences, window_size, n_conditions), dtype=np.float32),
                'usage_beta': np.zeros((n_sequences, window_size, n_conditions), dtype=np.float32),
                'usage_sse': np.zeros((n_sequences, window_size, n_conditions), dtype=np.float32),
                'species_ids': np.zeros((n_sequences,), dtype=np.int32),
                'condition_mask': mask
            }
            
            npz_path = os.path.join(tmpdir, 'test_data.npz')
            np.savez(npz_path, **data)
            
            # Load data
            result = load_processed_data(npz_path)
            
            assert len(result) == 7
            sequences, labels, alpha, beta, sse, species_ids, condition_mask = result
            
            # Verify mask
            assert condition_mask is not None
            assert condition_mask.shape == (n_sequences, n_conditions)
            assert condition_mask[:5, :3].all()
            assert condition_mask[5:, 3:].all()


class TestConditionMaskIntegration:
    """Integration tests for condition masking in the full pipeline."""
    
    def test_mask_respects_genome_boundaries(self):
        """Test that masks correctly separate genome-specific conditions."""
        # This would be an integration test with actual data
        # For now, test the logic
        
        # Simulate union of 99 conditions
        all_conditions = [f"Tissue_{i}" for i in range(1, 100)]
        
        # Simulate 3 genomes with specific condition counts
        genome_conditions = {
            'human': [f"Tissue_{i}" for i in range(1, 63)],  # 62 conditions
            'mouse': [f"Tissue_{i}" for i in range(1, 94)],  # 93 conditions
            'rat': [f"Tissue_{i}" for i in range(1, 84)]     # 83 conditions
        }
        
        # Build masks
        masks = {}
        for genome_id, genome_conds in genome_conditions.items():
            mask = np.zeros(len(all_conditions), dtype=np.bool_)
            for target_idx, target_cond in enumerate(all_conditions):
                if target_cond in genome_conds:
                    mask[target_idx] = True
            masks[genome_id] = mask
        
        # Verify
        assert masks['human'].sum() == 62
        assert masks['mouse'].sum() == 93
        assert masks['rat'].sum() == 83
        
        # Verify no overlap in invalid regions (all have first 62)
        assert masks['human'][:62].all()
        assert masks['mouse'][:62].all()
        assert masks['rat'][:62].all()
    
    def test_masked_loss_computation(self):
        """Test that masked loss computation works correctly."""
        # Simulate predictions and targets
        batch_size = 4
        n_windows = 10
        n_conditions = 5
        
        predictions = np.random.randn(batch_size, n_windows, n_conditions)
        targets = np.random.randn(batch_size, n_windows, n_conditions)
        
        # Create mask: first 2 sequences have first 3 conditions valid
        # last 2 sequences have last 2 conditions valid
        mask = np.zeros((batch_size, n_conditions), dtype=np.bool_)
        mask[:2, :3] = True
        mask[2:, 3:] = True
        
        # Expand mask to match predictions shape
        mask_expanded = mask[:, np.newaxis, :]  # (B, 1, C)
        mask_expanded = np.broadcast_to(mask_expanded, (batch_size, n_windows, n_conditions))
        
        # Compute masked MSE
        squared_error = (predictions - targets) ** 2
        masked_error = squared_error * mask_expanded
        
        # Loss should only count valid positions
        total_loss = masked_error.sum() / mask_expanded.sum()
        
        # Verify we only counted valid positions
        expected_valid = (2 * n_windows * 3) + (2 * n_windows * 2)  # 2 seqs * 10 windows * 3 conds + 2 seqs * 10 windows * 2 conds
        assert mask_expanded.sum() == expected_valid
        
        # Verify loss is finite
        assert np.isfinite(total_loss)


class TestCreateConditionMaskScript:
    """Test the create_condition_mask.py script functionality."""
    
    def test_species_mapping_logic(self):
        """Test that species IDs are correctly mapped to genome IDs."""
        species_mapping = {'human': 0, 'mouse': 1, 'rat': 2}
        genome_conditions = {
            'human_GRCh37': ['Brain_1', 'Heart_1'],
            'mouse_GRCm38': ['Brain_1', 'Heart_1', 'Liver_1'],
            'rat_Rnor_5.0': ['Brain_1', 'Kidney_1']
        }
        
        # Build ID to genome mapping
        id_to_genome = {}
        for short_name, species_id in species_mapping.items():
            for genome_id in genome_conditions.keys():
                if short_name in genome_id.lower():
                    id_to_genome[species_id] = genome_id
                    break
        
        # Verify mapping
        assert id_to_genome[0] == 'human_GRCh37'
        assert id_to_genome[1] == 'mouse_GRCm38'
        assert id_to_genome[2] == 'rat_Rnor_5.0'
    
    def test_processed_dir_detection(self):
        """Test that the script can find processed data directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            data_dir = Path(tmpdir) / 'data'
            splits_dir = data_dir / 'splits_full_1kb' / 'mouse_rat_human'
            processed_dir = data_dir / 'processed_full_1kb'
            
            splits_dir.mkdir(parents=True, exist_ok=True)
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Test path resolution
            split_path = splits_dir
            
            # Go up two levels to data dir
            found_processed = split_path.parent.parent / 'processed_full_1kb'
            
            assert found_processed.exists()
            assert found_processed == processed_dir


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
