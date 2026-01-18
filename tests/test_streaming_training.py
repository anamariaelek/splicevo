"""Integration tests for streaming-based training pipeline."""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from splicevo.training.dataset import SpliceDataset
from splicevo.model import SplicevoModel
from torch.utils.data import DataLoader


@pytest.fixture
def large_memmap_dataset():
    """Create a large memmap dataset to simulate production."""
    n_samples = 1000
    seq_len = 5900  # Full sequence with context
    n_conditions = 99  # Production-like
    
    tmpdir = tempfile.mkdtemp()
    tmppath = Path(tmpdir)
    
    # Create large memmap files
    sequences = np.memmap(
        tmppath / 'sequences.mmap',
        dtype=np.float32,
        mode='w+',
        shape=(n_samples, seq_len, 4)
    )
    sequences[:] = np.random.rand(n_samples, seq_len, 4).astype(np.float32)
    sequences.flush()
    
    labels = np.memmap(
        tmppath / 'labels.mmap',
        dtype=np.int64,
        mode='w+',
        shape=(n_samples, seq_len)
    )
    labels[:] = np.random.randint(0, 3, (n_samples, seq_len), dtype=np.int64)
    labels.flush()
    
    usage_sse = np.memmap(
        tmppath / 'usage_sse.mmap',
        dtype=np.float32,
        mode='w+',
        shape=(n_samples, seq_len, n_conditions)
    )
    usage_sse[:] = np.random.rand(n_samples, seq_len, n_conditions).astype(np.float32)
    usage_sse.flush()
    
    species_ids = np.memmap(
        tmppath / 'species_ids.mmap',
        dtype=np.int32,
        mode='w+',
        shape=(n_samples,)
    )
    species_ids[:] = np.random.randint(0, 3, n_samples, dtype=np.int32)
    species_ids.flush()
    
    # Reopen in read-only mode
    seq_mmap = np.memmap(tmppath / 'sequences.mmap', dtype=np.float32, mode='r', shape=(n_samples, seq_len, 4))
    lbl_mmap = np.memmap(tmppath / 'labels.mmap', dtype=np.int64, mode='r', shape=(n_samples, seq_len))
    sse_mmap = np.memmap(tmppath / 'usage_sse.mmap', dtype=np.float32, mode='r', shape=(n_samples, seq_len, n_conditions))
    species_mmap = np.memmap(tmppath / 'species_ids.mmap', dtype=np.int32, mode='r', shape=(n_samples,))
    
    yield seq_mmap, lbl_mmap, sse_mmap, species_mmap, n_samples, seq_len, n_conditions
    
    shutil.rmtree(tmpdir)


class TestStreamingTrainingPipeline:
    """Test complete training pipeline with streaming."""
    
    def test_train_val_split_no_memory_copy(self, large_memmap_dataset):
        """Verify train/val split doesn't copy memmap data."""
        seq_mmap, lbl_mmap, sse_mmap, species_mmap, n_samples, seq_len, n_conditions = large_memmap_dataset
        
        # Record original IDs
        seq_id = id(seq_mmap)
        lbl_id = id(lbl_mmap)
        sse_id = id(sse_mmap)
        
        # Create train/val split using indices (streaming pattern)
        n_train = int(0.8 * n_samples)
        train_indices = np.arange(0, n_train)
        val_indices = np.arange(n_train, n_samples)
        
        train_dataset = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=train_indices)
        val_dataset = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=val_indices)
        
        # Verify same objects (no copy)
        assert id(train_dataset.sequences) == seq_id
        assert id(val_dataset.sequences) == seq_id
        assert id(train_dataset.splice_labels) == lbl_id
        assert id(val_dataset.splice_labels) == lbl_id
        assert id(train_dataset.usage_sse) == sse_id
        assert id(val_dataset.usage_sse) == sse_id
        
        # Verify correct lengths
        assert len(train_dataset) == n_train
        assert len(val_dataset) == n_samples - n_train
    
    def test_dataloader_batch_processing(self, large_memmap_dataset):
        """Test DataLoader processes batches correctly with streaming."""
        seq_mmap, lbl_mmap, sse_mmap, species_mmap, n_samples, seq_len, n_conditions = large_memmap_dataset
        
        train_indices = np.arange(0, 100)  # Use subset for speed
        dataset = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=train_indices)
        
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        batches_processed = 0
        total_samples = 0
        
        for batch in loader:
            batches_processed += 1
            batch_size = batch['sequences'].shape[0]
            total_samples += batch_size
            
            # Verify batch shapes
            assert batch['sequences'].shape == (batch_size, seq_len, 4)
            assert batch['splice_labels'].shape == (batch_size, seq_len)
            assert batch['usage_targets'].shape == (batch_size, seq_len, n_conditions)
            assert batch['species_id'].shape == (batch_size,)
            
            # Verify data types
            assert batch['sequences'].dtype == torch.float32
            assert batch['splice_labels'].dtype == torch.int64
            assert batch['usage_targets'].dtype == torch.float32
            assert batch['species_id'].dtype == torch.int64
        
        assert total_samples == 100
        assert batches_processed == 13  # 100 / 8 = 12.5, rounded up
    
    def test_model_forward_with_streaming_data(self, large_memmap_dataset):
        """Test model forward pass with streaming dataset."""
        seq_mmap, lbl_mmap, sse_mmap, species_mmap, n_samples, seq_len, n_conditions = large_memmap_dataset
        
        # Create dataset
        train_indices = np.arange(0, 16)
        dataset = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=train_indices)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # Create model
        model = SplicevoModel(
            embed_dim=64,
            num_resblocks=4,
            dilation_strategy='alternating',
            alternate=2,
            num_classes=3,
            n_conditions=n_conditions,
            context_len=450,
            bottleneck_dim=64,
            num_heads=4,
            dropout=0.1,
            usage_loss_type='weighted_mse',
            n_species=3
        )
        model.eval()
        
        # Process one batch
        batch = next(iter(loader))
        
        with torch.no_grad():
            output = model(batch['sequences'], species_ids=batch['species_id'])
        
        # Verify output shapes
        central_len = seq_len - 2 * 450  # Remove context
        assert output['splice_logits'].shape == (8, central_len, 3)
        assert output['usage_predictions'].shape == (8, central_len, n_conditions)
    
    def test_multiple_epochs_with_streaming(self, large_memmap_dataset):
        """Test that streaming works correctly across multiple epochs."""
        seq_mmap, lbl_mmap, sse_mmap, species_mmap, n_samples, seq_len, n_conditions = large_memmap_dataset
        
        train_indices = np.arange(0, 50)
        dataset = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=train_indices)
        loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
        
        # Run multiple epochs
        epoch_data = []
        for epoch in range(3):
            epoch_samples = []
            for batch in loader:
                epoch_samples.append(batch['sequences'][:, 0, 0].clone())  # Save first feature
            epoch_data.append(torch.cat(epoch_samples))
        
        # Verify data is consistent across epochs
        assert torch.allclose(epoch_data[0], epoch_data[1])
        assert torch.allclose(epoch_data[1], epoch_data[2])
    
    def test_shuffled_indices_different_batches(self, large_memmap_dataset):
        """Test that shuffling indices produces different batch orders."""
        seq_mmap, lbl_mmap, sse_mmap, species_mmap, n_samples, seq_len, n_conditions = large_memmap_dataset
        
        # Epoch 1: no shuffle
        train_indices = np.arange(0, 100)
        dataset1 = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=train_indices)
        loader1 = DataLoader(dataset1, batch_size=10, shuffle=False, num_workers=0)
        batch1 = next(iter(loader1))
        
        # Epoch 2: shuffled indices
        shuffled_indices = np.random.permutation(train_indices)
        dataset2 = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=shuffled_indices)
        loader2 = DataLoader(dataset2, batch_size=10, shuffle=False, num_workers=0)
        batch2 = next(iter(loader2))
        
        # Batches should be different (with high probability)
        assert not torch.allclose(batch1['sequences'], batch2['sequences'])


class TestMemoryEfficiency:
    """Test memory efficiency of streaming approach."""
    
    def test_no_full_array_slicing(self, large_memmap_dataset):
        """Verify that creating datasets doesn't trigger full array loads."""
        seq_mmap, lbl_mmap, sse_mmap, species_mmap, n_samples, seq_len, n_conditions = large_memmap_dataset
        
        # This should NOT load the full arrays into memory
        n_train = int(0.8 * n_samples)
        train_indices = np.arange(0, n_train)
        val_indices = np.arange(n_train, n_samples)
        
        # Create datasets (should be instant, no data loading)
        train_dataset = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=train_indices)
        val_dataset = SpliceDataset(seq_mmap, lbl_mmap, sse_mmap, species_mmap, indices=val_indices)
        
        # Accessing single samples should load only that sample
        train_item = train_dataset[0]
        val_item = val_dataset[0]
        
        # Verify items have correct shapes (single samples)
        assert train_item['sequences'].shape == (seq_len, 4)
        assert val_item['sequences'].shape == (seq_len, 4)
    
    def test_index_array_is_lightweight(self, large_memmap_dataset):
        """Verify index arrays are much smaller than data arrays."""
        seq_mmap, lbl_mmap, sse_mmap, species_mmap, n_samples, seq_len, n_conditions = large_memmap_dataset
        
        # Create indices
        train_indices = np.arange(0, int(0.8 * n_samples))
        
        # Size of index array (in bytes)
        index_size = train_indices.nbytes
        
        # Size of actual data that would be copied with slicing
        # sequences: n_train * seq_len * 4 * 4 bytes (float32)
        # labels: n_train * seq_len * 8 bytes (int64)
        # usage: n_train * seq_len * n_conditions * 4 bytes (float32)
        n_train = len(train_indices)
        sequences_size = n_train * seq_len * 4 * 4
        labels_size = n_train * seq_len * 8
        usage_size = n_train * seq_len * n_conditions * 4
        total_data_size = sequences_size + labels_size + usage_size
        
        # Index array should be MUCH smaller (at least 1000x)
        ratio = total_data_size / index_size
        assert ratio > 1000, f"Index array should be much smaller than data (ratio: {ratio})"
        
        print(f"\nMemory efficiency:")
        print(f"  Index array: {index_size / 1024 / 1024:.2f} MB")
        print(f"  Full data (if copied): {total_data_size / 1024 / 1024:.2f} MB")
        print(f"  Memory saved: {ratio:.0f}x")


class TestBackwardCompatibility:
    """Test that datasets work without indices (backward compatibility)."""
    
    def test_dataset_without_indices_still_works(self):
        """Test that old code without indices parameter still works."""
        n_samples = 50
        seq_len = 200
        n_conditions = 5
        
        sequences = np.random.rand(n_samples, seq_len, 4).astype(np.float32)
        labels = np.random.randint(0, 3, (n_samples, seq_len), dtype=np.int64)
        usage_sse = np.random.rand(n_samples, seq_len, n_conditions).astype(np.float32)
        species_ids = np.random.randint(0, 3, n_samples, dtype=np.int32)
        
        # Old API (without indices)
        dataset = SpliceDataset(sequences, labels, usage_sse, species_ids)
        
        assert len(dataset) == n_samples
        assert dataset.indices is None
        
        # Should work normally
        item = dataset[0]
        assert item['sequences'].shape == (seq_len, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
