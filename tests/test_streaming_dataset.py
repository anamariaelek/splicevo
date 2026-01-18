"""Tests for streaming dataset with index-based memmap access."""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
from splicevo.training.dataset import SpliceDataset


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n_samples = 100
    seq_len = 200
    n_conditions = 5
    
    sequences = np.random.rand(n_samples, seq_len, 4).astype(np.float32)
    labels = np.random.randint(0, 3, (n_samples, seq_len), dtype=np.int64)
    usage_sse = np.random.rand(n_samples, seq_len, n_conditions).astype(np.float32)
    species_ids = np.random.randint(0, 3, n_samples, dtype=np.int32)
    
    return sequences, labels, usage_sse, species_ids


@pytest.fixture
def memmap_data(sample_data):
    """Create memory-mapped files for testing streaming."""
    sequences, labels, usage_sse, species_ids = sample_data
    
    # Create temporary directory
    tmpdir = tempfile.mkdtemp()
    tmppath = Path(tmpdir)
    
    # Create memmap files
    seq_mmap = np.memmap(
        tmppath / 'sequences.mmap',
        dtype=np.float32,
        mode='w+',
        shape=sequences.shape
    )
    seq_mmap[:] = sequences[:]
    seq_mmap.flush()
    
    lbl_mmap = np.memmap(
        tmppath / 'labels.mmap',
        dtype=np.int64,
        mode='w+',
        shape=labels.shape
    )
    lbl_mmap[:] = labels[:]
    lbl_mmap.flush()
    
    sse_mmap = np.memmap(
        tmppath / 'usage_sse.mmap',
        dtype=np.float32,
        mode='w+',
        shape=usage_sse.shape
    )
    sse_mmap[:] = usage_sse[:]
    sse_mmap.flush()
    
    species_mmap = np.memmap(
        tmppath / 'species_ids.mmap',
        dtype=np.int32,
        mode='w+',
        shape=species_ids.shape
    )
    species_mmap[:] = species_ids[:]
    species_mmap.flush()
    
    # Reopen in read-only mode (like production)
    seq_mmap_ro = np.memmap(tmppath / 'sequences.mmap', dtype=np.float32, mode='r', shape=sequences.shape)
    lbl_mmap_ro = np.memmap(tmppath / 'labels.mmap', dtype=np.int64, mode='r', shape=labels.shape)
    sse_mmap_ro = np.memmap(tmppath / 'usage_sse.mmap', dtype=np.float32, mode='r', shape=usage_sse.shape)
    species_mmap_ro = np.memmap(tmppath / 'species_ids.mmap', dtype=np.int32, mode='r', shape=species_ids.shape)
    
    yield seq_mmap_ro, lbl_mmap_ro, sse_mmap_ro, species_mmap_ro, sequences, labels, usage_sse, species_ids
    
    # Cleanup
    shutil.rmtree(tmpdir)


class TestSpliceDatasetBasic:
    """Test basic dataset functionality without indices."""
    
    def test_dataset_creation(self, sample_data):
        """Test creating dataset with regular numpy arrays."""
        sequences, labels, usage_sse, species_ids = sample_data
        
        dataset = SpliceDataset(sequences, labels, usage_sse, species_ids)
        
        assert len(dataset) == len(sequences)
        assert dataset.sequences is sequences
        assert dataset.splice_labels is labels
        assert dataset.usage_sse is usage_sse
        assert dataset.species_ids is species_ids
    
    def test_dataset_getitem(self, sample_data):
        """Test retrieving items from dataset."""
        sequences, labels, usage_sse, species_ids = sample_data
        
        dataset = SpliceDataset(sequences, labels, usage_sse, species_ids)
        
        # Get first item
        item = dataset[0]
        
        assert 'sequences' in item
        assert 'splice_labels' in item
        assert 'usage_targets' in item
        assert 'species_id' in item
        
        # Check shapes
        assert item['sequences'].shape == (sequences.shape[1], sequences.shape[2])
        assert item['splice_labels'].shape == (labels.shape[1],)
        assert item['usage_targets'].shape == (usage_sse.shape[1], usage_sse.shape[2])
        assert item['species_id'].shape == ()
        
        # Check types
        assert isinstance(item['sequences'], torch.Tensor)
        assert isinstance(item['splice_labels'], torch.Tensor)
        assert isinstance(item['usage_targets'], torch.Tensor)
        assert isinstance(item['species_id'], torch.Tensor)
    
    def test_dataset_without_usage(self, sample_data):
        """Test dataset without usage data."""
        sequences, labels, _, species_ids = sample_data
        
        dataset = SpliceDataset(sequences, labels, None, species_ids)
        
        item = dataset[0]
        
        # Should have dummy usage tensor
        assert 'usage_targets' in item
        assert item['usage_targets'].shape[1] == 1  # Dummy dimension
    
    def test_dataset_without_species(self, sample_data):
        """Test dataset without species IDs."""
        sequences, labels, usage_sse, _ = sample_data
        
        dataset = SpliceDataset(sequences, labels, usage_sse, None)
        
        item = dataset[0]
        
        # Should have default species ID of 0
        assert item['species_id'].item() == 0


class TestSpliceDatasetWithIndices:
    """Test index-based streaming functionality."""
    
    def test_dataset_with_indices(self, sample_data):
        """Test creating dataset with index subset."""
        sequences, labels, usage_sse, species_ids = sample_data
        
        # Create train indices (first 80%)
        train_indices = np.arange(0, 80)
        
        dataset = SpliceDataset(sequences, labels, usage_sse, species_ids, indices=train_indices)
        
        assert len(dataset) == 80
        assert dataset.indices is train_indices
    
    def test_dataset_indices_access(self, sample_data):
        """Test that indices correctly map to actual data."""
        sequences, labels, usage_sse, species_ids = sample_data
        
        # Create indices for samples 10-20
        indices = np.arange(10, 20)
        
        dataset = SpliceDataset(sequences, labels, usage_sse, species_ids, indices=indices)
        
        # Dataset index 0 should map to actual index 10
        item = dataset[0]
        expected_sequences = sequences[10]
        expected_labels = labels[10]
        expected_species = species_ids[10]
        
        assert torch.allclose(item['sequences'], torch.from_numpy(expected_sequences))
        assert torch.equal(item['splice_labels'], torch.from_numpy(expected_labels))
        assert item['species_id'].item() == expected_species
    
    def test_train_val_split_with_indices(self, sample_data):
        """Test train/val split using indices (streaming pattern)."""
        sequences, labels, usage_sse, species_ids = sample_data
        n_samples = len(sequences)
        n_train = int(0.8 * n_samples)
        
        # Create index arrays (no data slicing!)
        train_indices = np.arange(0, n_train)
        val_indices = np.arange(n_train, n_samples)
        
        train_dataset = SpliceDataset(sequences, labels, usage_sse, species_ids, indices=train_indices)
        val_dataset = SpliceDataset(sequences, labels, usage_sse, species_ids, indices=val_indices)
        
        assert len(train_dataset) == n_train
        assert len(val_dataset) == n_samples - n_train
        
        # Verify no overlap between train and val
        train_item = train_dataset[0]
        val_item = val_dataset[0]
        
        # These should be different samples
        assert not torch.equal(train_item['sequences'], val_item['sequences'])
    
    def test_indices_out_of_order(self, sample_data):
        """Test that indices can be in any order (for shuffling)."""
        sequences, labels, usage_sse, species_ids = sample_data
        
        # Shuffled indices
        indices = np.array([50, 10, 99, 0, 42])
        
        dataset = SpliceDataset(sequences, labels, usage_sse, species_ids, indices=indices)
        
        assert len(dataset) == 5
        
        # Check that dataset[0] returns data from actual index 50
        item = dataset[0]
        expected_sequences = sequences[50]
        assert torch.allclose(item['sequences'], torch.from_numpy(expected_sequences))
        
        # Check that dataset[1] returns data from actual index 10
        item = dataset[1]
        expected_sequences = sequences[10]
        assert torch.allclose(item['sequences'], torch.from_numpy(expected_sequences))


class TestMemoryMappedStreaming:
    """Test streaming from memory-mapped files."""
    
    def test_memmap_basic_access(self, memmap_data):
        """Test basic access to memmap files."""
        seq_mmap, lbl_mmap, sse_mmap, species_mmap, *_ = memmap_data
        
        dataset = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap)
        
        assert len(dataset) == len(seq_mmap)
        
        # Access should work
        item = dataset[0]
        assert item['sequences'].shape[0] == seq_mmap.shape[1]
    
    def test_memmap_streaming_with_indices(self, memmap_data):
        """Test streaming pattern with memmap (no full array loading)."""
        seq_mmap, lbl_mmap, sse_mmap, species_mmap, sequences, labels, usage_sse, species_ids = memmap_data
        
        n_samples = len(seq_mmap)
        n_train = int(0.8 * n_samples)
        
        # Create index arrays (streaming pattern)
        train_indices = np.arange(0, n_train)
        val_indices = np.arange(n_train, n_samples)
        
        train_dataset = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=train_indices)
        val_dataset = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=val_indices)
        
        # Verify train dataset accesses correct data
        train_item = train_dataset[0]
        expected_seq = sequences[0]  # train_indices[0] = 0
        assert torch.allclose(train_item['sequences'], torch.from_numpy(expected_seq))
        
        # Verify val dataset accesses correct data
        val_item = val_dataset[0]
        expected_seq = sequences[n_train]  # val_indices[0] = n_train
        assert torch.allclose(val_item['sequences'], torch.from_numpy(expected_seq))
    
    def test_memmap_multiple_epochs(self, memmap_data):
        """Test that memmap can be accessed multiple times (multiple epochs)."""
        seq_mmap, lbl_mmap, sse_mmap, species_mmap, sequences, *_ = memmap_data
        
        indices = np.arange(0, 10)
        dataset = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=indices)
        
        # Access same sample multiple times
        item1 = dataset[5]
        item2 = dataset[5]
        item3 = dataset[5]
        
        # Should get identical data each time
        assert torch.equal(item1['sequences'], item2['sequences'])
        assert torch.equal(item2['sequences'], item3['sequences'])
        assert torch.equal(item1['splice_labels'], item2['splice_labels'])
    
    def test_memmap_no_data_copy_on_split(self, memmap_data):
        """Test that creating train/val splits doesn't copy memmap data."""
        seq_mmap, lbl_mmap, sse_mmap, species_mmap, *_ = memmap_data
        
        # Record original memmap object
        original_seq_id = id(seq_mmap)
        
        n_samples = len(seq_mmap)
        n_train = int(0.8 * n_samples)
        
        train_indices = np.arange(0, n_train)
        val_indices = np.arange(n_train, n_samples)
        
        train_dataset = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=train_indices)
        val_dataset = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=val_indices)
        
        # Both datasets should reference the SAME memmap object (no copy)
        assert id(train_dataset.sequences) == original_seq_id
        assert id(val_dataset.sequences) == original_seq_id
        assert train_dataset.sequences is val_dataset.sequences


class TestDataLoaderIntegration:
    """Test integration with PyTorch DataLoader."""
    
    def test_dataloader_with_streaming(self, memmap_data):
        """Test DataLoader with streaming dataset."""
        from torch.utils.data import DataLoader
        
        seq_mmap, lbl_mmap, sse_mmap, species_mmap, *_ = memmap_data
        
        indices = np.arange(0, 50)
        dataset = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=indices)
        
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        batch = next(iter(loader))
        
        assert batch['sequences'].shape[0] == 8  # batch size
        assert batch['splice_labels'].shape[0] == 8
        assert batch['usage_targets'].shape[0] == 8
        assert batch['species_id'].shape[0] == 8
    
    def test_dataloader_iteration(self, sample_data):
        """Test iterating through entire dataset with DataLoader."""
        from torch.utils.data import DataLoader
        
        sequences, labels, usage_sse, species_ids = sample_data
        indices = np.arange(0, 50)
        
        dataset = SpliceDataset(sequences, labels, usage_sse, species_ids, indices=indices)
        loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
        
        total_samples = 0
        for batch in loader:
            total_samples += batch['sequences'].shape[0]
        
        assert total_samples == 50


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_indices(self, sample_data):
        """Test dataset with empty indices array."""
        sequences, labels, usage_sse, species_ids = sample_data
        
        indices = np.array([], dtype=np.int64)
        dataset = SpliceDataset(sequences, labels, usage_sse, species_ids, indices=indices)
        
        assert len(dataset) == 0
    
    def test_single_sample_indices(self, sample_data):
        """Test dataset with single sample."""
        sequences, labels, usage_sse, species_ids = sample_data
        
        indices = np.array([42])
        dataset = SpliceDataset(sequences, labels, usage_sse, species_ids, indices=indices)
        
        assert len(dataset) == 1
        item = dataset[0]
        assert torch.allclose(item['sequences'], torch.from_numpy(sequences[42]))
    
    def test_duplicate_indices(self, sample_data):
        """Test that duplicate indices work (for oversampling)."""
        sequences, labels, usage_sse, species_ids = sample_data
        
        # Indices with duplicates
        indices = np.array([0, 0, 1, 1, 1, 2])
        dataset = SpliceDataset(sequences, labels, usage_sse, species_ids, indices=indices)
        
        assert len(dataset) == 6
        
        # dataset[0] and dataset[1] should be identical
        item0 = dataset[0]
        item1 = dataset[1]
        assert torch.equal(item0['sequences'], item1['sequences'])
        assert torch.equal(item0['splice_labels'], item1['splice_labels'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
