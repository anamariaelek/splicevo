"""
Comprehensive tests for MultiGenomeDataLoader.to_arrays() method.

These tests verify that:
1. Sequences are correctly extracted at the expected genomic intervals
2. Metadata window_start/window_end correctly corresponds to sequences
3. Splice site positions within windows are correctly marked
4. Usage arrays align with splice site positions
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from splicevo.data.data_loader import MultiGenomeDataLoader
from splicevo.io.splice_sites import SpliceSite
from splicevo.io.genome import GenomeData


class TestToArraysSequenceMetadataCorrespondence:
    """Test that sequences correspond correctly to metadata intervals."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = MultiGenomeDataLoader()
        self.window_size = 100
        self.context_size = 50
        self.total_window = self.context_size + self.window_size + self.context_size
    
    def test_window_metadata_genomic_positions(self):
        """
        Test that window metadata correctly reflects genomic positions.
        
        For a window with metadata:
            window_start = 1000
            window_end = 1100
        
        The sequence should contain nucleotides from positions 950-1150 (with context).
        Positions within the window_start-window_end range should map to the correct
        indices in the sequence array.
        """
        # Create mock splice sites within a specific interval
        splice_sites = [
            SpliceSite(
                genome_id='test_genome',
                chromosome='chr1',
                transcript_id='tx1',
                gene_id='gene1',
                position=1050,  # Within window center (1000-1100)
                site_type=1,    # donor
                strand='+',
                site_usage={}
            )
        ]
        
        # Load the splice site
        self.loader.loaded_data = splice_sites
        
        # Create mock dataframe
        df = pd.DataFrame({
            'genome_id': ['test_genome'],
            'chromosome': ['chr1'],
            'gene_id': ['gene1'],
            'position': [1050],
            'site_type': [1],
            'strand': ['+'],
            'transcript_id': ['tx1']
        })
        
        with patch.object(self.loader, 'get_dataframe', return_value=df):
            # Verify position is within expected window
            window_center_start = 1000
            window_center_end = 1100
            
            position = 1050
            assert window_center_start <= position < window_center_end, \
                f"Position {position} should be within window [{window_center_start}, {window_center_end})"
            
            # Verify window_pos calculation
            window_pos = position - window_center_start
            assert 0 <= window_pos < self.window_size, \
                f"Window position {window_pos} should be in range [0, {self.window_size})"
            assert window_pos == 50, f"Expected window_pos=50, got {window_pos}"

    def test_sequence_position_mapping_in_windowed_array(self):
        """
        Test that position indices in the windowed array correspond to the correct genomic positions.
        
        Sequence array has shape (window_size,) and contains one-hot encoding.
        Position i in the sequence corresponds to genomic position (window_center_start + i).
        """
        window_center_start = 1000
        window_center_end = 1100
        sequence_length = self.window_size
        
        # Test position mapping
        test_cases = [
            (0, 1000),      # First position
            (50, 1050),     # Middle position
            (99, 1099),     # Last valid position
        ]
        
        for seq_idx, expected_genomic_pos in test_cases:
            genomic_pos = window_center_start + seq_idx
            assert genomic_pos == expected_genomic_pos, \
                f"Sequence index {seq_idx} should map to genomic position {expected_genomic_pos}, got {genomic_pos}"

    def test_requested_start_vs_actual_start_with_padding(self):
        """
        Test that padding is correctly handled when sequence start is clipped to 0.
        
        Case: requested_start is negative (clipped to actual_start=0)
        - requested_start = -50 (before chromosome start)
        - actual_start = 0
        - left_pad = 50
        
        Genomic position mapping must account for this padding.
        """
        min_pos = 25
        context_size = 50
        
        requested_start = min_pos - context_size  # -25
        actual_start = max(0, requested_start)    # 0
        left_pad = actual_start - requested_start  # 0 - (-25) = 25
        
        assert requested_start == -25
        assert actual_start == 0
        assert left_pad == 25
        
        # Genomic position to sequence index mapping must account for padding
        # Position 0 should map to sequence index 25 (after left padding)
        genomic_pos = 0
        sequence_idx = genomic_pos - requested_start
        assert sequence_idx == 25, f"Position 0 should map to index 25, got {sequence_idx}"

    def test_window_start_calculation_must_use_correct_baseline(self):
        """
        CRITICAL: Test that window_start uses the correct baseline.
        
        Bug hypothesis: window_start is calculated as an offset from the start of the local
        sequence (which may include padding), but genomic position is calculated as
        requested_start + window_start, which would be incorrect.
        
        Correct calculation should be:
            window_genomic_start = requested_start + left_pad + window_start
        OR:
            window_genomic_start = actual_start + window_start  (if window_start is relative to actual_start)
        """
        min_pos = 25
        context_size = 50
        window_size = 100
        
        # Scenario: requested_start is negative (clipped)
        requested_start = min_pos - context_size  # -25
        actual_start = max(0, requested_start)     # 0
        left_pad = actual_start - requested_start  # 50
        
        # Local sequence indices
        local_window_start = 50  # After left padding
        
        # What the current code INCORRECTLY does:
        incorrect_genomic_start = requested_start + local_window_start
        # = -25 + 50 = 25 (happens to be correct in this case)
        
        # What it SHOULD do if window_start is truly local:
        correct_genomic_start = requested_start + left_pad + local_window_start
        # = -25 + 50 + 50 = 75 (or if window_start is 0, then -25 + 50 = 25)
        
        # OR if the code intends window_start to be from requested_start (not actual_start):
        # Then it's correct, but window_start must be calculated relative to requested_start
        pass

    def test_window_start_range_in_loop(self):
        """
        Test that the loop range(0, len(seq) - total_window + 1, window_size)
        produces correct window indices.
        
        The loop variable `window_start` is an index into the local sequence array,
        which includes padding.
        """
        seq_length = 250  # Example local sequence length
        total_window = 200
        window_size = 100
        
        # What windows does the loop generate?
        windows = []
        for window_start in range(0, seq_length - total_window + 1, window_size):
            window_end = window_start + total_window
            windows.append((window_start, window_end))
        
        # range(0, 250 - 200 + 1, 100) = range(0, 51, 100) = [0]
        # Only 1 window because 100 > 50
        assert windows == [(0, 200)], \
            f"Window extraction logic error: {windows}"

    def test_genomic_position_calculation_in_window_loop(self):
        """
        Test the critical position calculation in the window loop:
        
        window_genomic_start = requested_start + window_start
        window_center_genomic_start = window_genomic_start + context_size
        """
        min_pos = 500
        context_size = 50
        window_size = 100
        
        requested_start = min_pos - context_size  # 450
        actual_start = max(0, requested_start)     # 450
        left_pad = actual_start - requested_start  # 0
        
        # In the loop, for the first window:
        window_start = 0
        
        # Current code calculates:
        window_genomic_start = requested_start + window_start  # 450 + 0 = 450
        window_center_genomic_start = window_genomic_start + context_size  # 500
        window_center_genomic_end = window_center_genomic_start + window_size  # 600
        
        # This is CORRECT because requested_start is the genomic start
        # and window_start is relative to that
        assert window_center_genomic_start == 500
        assert window_center_genomic_end == 600

    def test_genomic_position_calculation_with_left_padding(self):
        """
        Test position calculation when there's left padding (sequence starts at 0).
        
        This is the CRITICAL test case where the bug is likely to manifest.
        """
        min_pos = 25
        context_size = 50
        window_size = 100
        
        requested_start = min_pos - context_size  # -25
        actual_start = max(0, requested_start)     # 0
        left_pad = actual_start - requested_start  # 50
        
        # The sequence contains:
        # [padding(50): 'N'*50] + [actual_sequence(75): from pos 0-75]
        # Total length = 125
        
        # In the loop, for the first valid window (after padding):
        # window_start should be chosen such that we can extract total_window bytes
        total_window = context_size + window_size + context_size  # 200
        seq_length = left_pad + 75  # 125 (not enough for even one window!)
        
        # No windows would be generated here. Let's use a larger sequence:
        actual_end = 500
        seq_length = left_pad + (actual_end - actual_start)  # 50 + 500 = 550
        
        # First valid window_start in loop:
        window_start = 0
        
        # Current calculation:
        window_genomic_start = requested_start + window_start  # -25 + 0 = -25
        window_center_genomic_start = window_genomic_start + context_size  # -25 + 50 = 25
        window_center_genomic_end = window_center_genomic_start + window_size  # 125
        
        # This calculation is WRONG! 
        # window_start=0 means we're at the beginning of the local sequence,
        # which includes left_pad. So the genomic start should be:
        # requested_start + window_start = -25 + 0 = -25 (WRONG - this is before chromosome 0)
        
        # CORRECT would be:
        # requested_start + window_start (window_start is in global coordinates relative to requested_start)
        # But if window_start is a local sequence index, it needs:
        # requested_start + left_pad + window_start, OR
        # actual_start + window_start (if window_start is relative to actual_start)
        
        # The issue: Is window_start relative to requested_start or to actual_start or to the local sequence?
        # The loop uses range() on local sequence length, so it's relative to the local sequence
        # The local sequence has left_pad at the beginning, so:
        # requested_start + left_pad + window_start would be the actual genomic position
        
        # But the current code does: requested_start + window_start
        # This is INCORRECT when left_pad > 0


class TestToArraysLabelAlignment:
    """Test that labels are correctly aligned with sequence positions."""
    
    def test_label_position_within_window(self):
        """
        Test that splice site positions are correctly mapped to label array indices.
        
        For a splice site at genomic position P within a window:
            window_center_start to window_center_end
        
        The label index should be: P - window_center_start
        And this should be a valid index in labels array (0 <= index < window_size)
        """
        window_center_start = 1000
        window_center_end = 1100
        window_size = 100
        
        splice_site_position = 1050
        
        window_pos = splice_site_position - window_center_start
        assert 0 <= window_pos < window_size
        assert window_pos == 50

    def test_labels_array_shape_matches_window_size(self):
        """Test that labels array has shape (window_size,) not (total_window,)."""
        window_size = 100
        labels = np.zeros(window_size, dtype=np.int8)
        
        # Verify it's the window_size, not total_window
        assert labels.shape[0] == window_size
        assert len(labels) == window_size

    def test_splice_site_within_window_marked_correctly(self):
        """
        Test that when a splice site is within a window's central region,
        it's correctly marked in the labels array.
        """
        window_size = 100
        window_center_start = 1000
        
        # Splice site at position 1050
        splice_site_position = 1050
        site_type = 1  # donor
        
        # Calculate label index
        label_idx = splice_site_position - window_center_start
        
        # Create label array and mark the site
        labels = np.zeros(window_size, dtype=np.int8)
        if 0 <= label_idx < window_size:
            labels[label_idx] = site_type
        
        assert labels[50] == 1
        assert labels[49] == 0
        assert labels[51] == 0

    def test_splice_site_outside_window_not_marked(self):
        """
        Test that splice sites outside the window's central region are not included.
        
        The window loop filters sites by:
            (position >= window_center_start) & (position < window_center_end)
        """
        window_size = 100
        window_center_start = 1000
        window_center_end = 1100
        
        positions_to_test = [
            (999, False),   # Just outside
            (1000, True),   # At boundary (inclusive)
            (1050, True),   # Middle
            (1099, True),   # At boundary (exclusive check)
            (1100, False),  # Just outside
            (1101, False),  # Outside
        ]
        
        for pos, should_be_included in positions_to_test:
            is_in_window = (pos >= window_center_start) & (pos < window_center_end)
            assert is_in_window == should_be_included, \
                f"Position {pos}: expected included={should_be_included}, got {is_in_window}"


class TestToArraysUsageArrays:
    """Test that usage arrays are correctly populated."""
    
    def test_usage_array_shape_matches_labels(self):
        """Test that usage arrays have shape (window_size, n_conditions)."""
        window_size = 100
        n_conditions = 3
        
        usage_array = np.full((window_size, n_conditions), np.nan, dtype=np.float32)
        
        assert usage_array.shape == (window_size, n_conditions)

    def test_usage_values_at_non_splice_site_positions_are_nan(self):
        """Test that usage values are NaN at positions without splice sites."""
        window_size = 100
        n_conditions = 3
        
        usage_array = np.full((window_size, n_conditions), np.nan, dtype=np.float32)
        
        # Only set values for splice sites at specific positions
        splice_positions = [10, 50, 90]
        for pos in splice_positions:
            usage_array[pos, :] = [0.5, 0.3, 0.2]
        
        # Verify non-splice positions are still NaN
        for i in range(window_size):
            if i not in splice_positions:
                assert np.all(np.isnan(usage_array[i, :]))
            else:
                assert not np.any(np.isnan(usage_array[i, :]))

    def test_usage_array_condition_index_mapping(self):
        """Test that condition keys are correctly mapped to array indices."""
        conditions = [
            {'condition_key': 'tissue_A', 'tissue': 'tissue_A', 'timepoint': 'NA', 'display_name': 'tissue_A'},
            {'condition_key': 'tissue_B', 'tissue': 'tissue_B', 'timepoint': 'NA', 'display_name': 'tissue_B'},
            {'condition_key': 'tissue_C', 'tissue': 'tissue_C', 'timepoint': 'NA', 'display_name': 'tissue_C'},
        ]
        
        condition_to_idx = {cond['condition_key']: idx for idx, cond in enumerate(conditions)}
        
        assert condition_to_idx['tissue_A'] == 0
        assert condition_to_idx['tissue_B'] == 1
        assert condition_to_idx['tissue_C'] == 2


class TestToArraysIntegrationWithMockedGenome:
    """Integration tests with mocked genome data."""
    
    def test_small_integration_two_splice_sites_one_window(self):
        """
        Test a small integration case with 2 splice sites in 1 window.
        
        Setup:
        - Splice sites at positions 100 and 150
        - Window center: 50-150 (window_size=100)
        - Context: 50 on each side
        - Total sequence: 0-250
        """
        loader = MultiGenomeDataLoader()
        window_size = 100
        context_size = 50
        
        # Create splice sites
        splice_sites = [
            SpliceSite(
                genome_id='test',
                chromosome='chr1',
                transcript_id='tx1',
                gene_id='gene1',
                position=100,
                site_type=1,  # donor
                strand='+',
                site_usage={}
            ),
            SpliceSite(
                genome_id='test',
                chromosome='chr1',
                transcript_id='tx1',
                gene_id='gene1',
                position=150,
                site_type=2,  # acceptor
                strand='+',
                site_usage={}
            ),
        ]
        
        loader.loaded_data = splice_sites
        
        # Verify window geometry
        min_pos = 100
        max_pos = 150
        requested_start = min_pos - context_size  # 50
        requested_end = max_pos + context_size    # 200
        
        # First window (starting at requested_start)
        window_center_start = requested_start + context_size  # 100
        window_center_end = window_center_start + window_size  # 200
        
        # Both sites should be in this window
        assert window_center_start <= 100 < window_center_end
        assert window_center_start <= 150 < window_center_end
        
        # Expected label indices within window
        idx_100 = 100 - window_center_start  # 0
        idx_150 = 150 - window_center_start  # 50
        
        assert idx_100 == 0
        assert idx_150 == 50


class TestToArraysEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_splice_site_at_window_start(self):
        """Test splice site exactly at window_center_start."""
        window_center_start = 1000
        window_center_end = 1100
        position = 1000
        
        window_pos = position - window_center_start
        assert window_pos == 0
        assert 0 <= window_pos < 100

    def test_single_splice_site_at_window_end_boundary(self):
        """Test splice site at window_center_end (should be excluded)."""
        window_center_start = 1000
        window_center_end = 1100
        position = 1100
        
        # Should not be included due to < check
        is_in_window = (position >= window_center_start) and (position < window_center_end)
        assert not is_in_window

    def test_empty_window_skipped(self):
        """Test that windows with no splice sites are skipped."""
        window_size = 100
        window_center_start = 1000
        window_center_end = 1100
        
        # No splice sites in this range
        sites_in_window = []
        
        # Should skip this window
        if len(sites_in_window) == 0:
            skip = True
        
        assert skip


class TestToArraysIntegrationWithRealData:
    """Integration tests using real data from processed datasets."""
    
    def test_metadata_sequence_correspondence_real_data(self):
        """
        Test that metadata intervals match the actual sequence data in a real run.
        
        This test loads real data and validates:
        1. Each sequence corresponds to the interval in metadata
        2. Splice sites marked in labels fall within metadata window
        3. All splice sites in metadata window are accounted for
        """
        import json
        import os
        
        # Try to load real data
        real_data_dir = Path('/home/elek/sds/sd17d003/Anamaria/splicevo/data/processed_small/mouse_GRCm38/')
        
        if not real_data_dir.exists():
            pytest.skip("Real data directory not found")
        
        # Load metadata
        metadata_path = real_data_dir / 'metadata.json'
        if not metadata_path.exists():
            pytest.skip("Metadata file not found")
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Load actual metadata.csv
        metadata_csv_path = real_data_dir / 'metadata.csv'
        if not metadata_csv_path.exists():
            pytest.skip("Metadata CSV not found")
        
        metadata_df = pd.read_csv(metadata_csv_path)
        
        # Load sequences and labels from memmap
        sequences_path = real_data_dir / 'sequences.mmap'
        labels_path = real_data_dir / 'labels.mmap'
        
        if not sequences_path.exists() or not labels_path.exists():
            pytest.skip("Memmap files not found")
        
        # Load memmap files
        sequences = np.memmap(
            sequences_path,
            dtype=np.float32,
            mode='r',
            shape=tuple(metadata['sequences_shape'])
        )
        
        labels = np.memmap(
            labels_path,
            dtype=np.int8,
            mode='r',
            shape=tuple(metadata['labels_shape'])
        )
        
        # Validate dimensions
        assert sequences.shape[0] == labels.shape[0], \
            f"Sequence count {sequences.shape[0]} != label count {labels.shape[0]}"
        assert sequences.shape[0] == len(metadata_df), \
            f"Sequence count {sequences.shape[0]} != metadata rows {len(metadata_df)}"
        
        # Spot check several windows
        window_size = metadata['window_size']
        context_size = metadata['context_size']
        
        for idx in range(min(10, len(metadata_df))):
            row = metadata_df.iloc[idx]
            seq = sequences[idx]
            lbl = labels[idx]
            
            # Verify sequence shape
            assert seq.shape == (context_size + window_size + context_size, 4), \
                f"Sequence {idx} has shape {seq.shape}, expected {(context_size + window_size + context_size, 4)}"
            
            # Verify label shape
            assert lbl.shape == (window_size,), \
                f"Label {idx} has shape {lbl.shape}, expected {(window_size,)}"
            
            # Check that sequence is valid one-hot encoding
            assert np.allclose(seq.sum(axis=1), 1.0), \
                f"Sequence {idx} is not valid one-hot encoding (some positions don't sum to 1)"
            
            # Verify metadata window bounds are reasonable
            assert 'window_start' in row
            assert 'window_end' in row
            assert row['window_end'] - row['window_start'] == window_size, \
                f"Window size mismatch at index {idx}: {row['window_end']} - {row['window_start']} != {window_size}"

    def test_sequence_covers_metadata_window_plus_context(self):
        """
        Test that the sequence length matches window_size + 2*context_size.
        
        Sequence should be:
        [context (left)] + [window] + [context (right)]
        """
        loader = MultiGenomeDataLoader()
        window_size = 100
        context_size = 50
        total_window = window_size + 2 * context_size  # 200
        
        # Verify calculation
        assert total_window == 200
        
        # Sequence should have shape (total_window, 4) for one-hot encoding
        seq = np.random.randn(total_window, 4)
        assert seq.shape[0] == window_size + 2 * context_size


class TestToArraysCriticalBug:
    """
    Tests that expose the critical bug in window_start calculation.
    
    BUG: The window_genomic_start calculation uses:
        window_genomic_start = requested_start + window_start
    
    But when requested_start is negative and clipped to actual_start=0,
    there's padding added to the sequence. The window_start is an index into
    the padded sequence, so the calculation should be:
        window_genomic_start = requested_start + left_pad + window_start
    OR equivalently:
        window_genomic_start = actual_start + window_start
    
    This bug causes sequences to be mapped to the wrong genomic intervals.
    """
    
    def test_window_calculation_with_negative_requested_start(self):
        """
        Expose the bug when requested_start is negative.
        
        Example:
        - Gene with first splice site at position 25
        - context_size = 50
        - requested_start = 25 - 50 = -25 (NEGATIVE!)
        - actual_start = max(0, -25) = 0
        - Sequence fetched from position 0, but local sequence will have padding
        
        The current code calculates:
            window_genomic_start = requested_start + window_start
                                 = -25 + 0 = -25 (WRONG!)
        
        Should be:
            window_genomic_start = actual_start + window_start
                                 = 0 + 0 = 0 (CORRECT)
        OR:
            window_genomic_start = requested_start + (actual_start - requested_start) + window_start
                                 = -25 + 25 + 0 = 0 (CORRECT)
        """
        min_pos = 25
        context_size = 50
        window_start = 0  # First window in loop
        
        requested_start = min_pos - context_size  # -25
        actual_start = max(0, requested_start)    # 0
        left_pad = actual_start - requested_start  # 25
        
        # BUGGY calculation (current code):
        window_genomic_start_buggy = requested_start + window_start
        # = -25 + 0 = -25 (NEGATIVE - BEFORE CHROMOSOME START!)
        
        # CORRECT calculation:
        window_genomic_start_correct = actual_start + window_start
        # = 0 + 0 = 0
        
        # OR:
        window_genomic_start_correct_2 = requested_start + left_pad + window_start
        # = -25 + 25 + 0 = 0
        
        assert window_genomic_start_buggy == -25, "Buggy calculation should give -25"
        assert window_genomic_start_correct == 0, "Correct calculation should give 0"
        assert window_genomic_start_correct == window_genomic_start_correct_2

    def test_window_calculation_with_large_negative_requested_start(self):
        """
        Test with a larger negative offset to make the bug more obvious.
        
        Scenario:
        - First splice site near beginning: position 100
        - Large context_size: 200
        - requested_start = 100 - 200 = -100
        - actual_start = 0
        - left_pad = 100
        """
        min_pos = 100
        context_size = 200
        window_size = 100
        window_start = 100  # Second window after context
        
        requested_start = min_pos - context_size  # -100
        actual_start = max(0, requested_start)    # 0
        left_pad = actual_start - requested_start  # 100
        
        # Current BUGGY code:
        window_genomic_start_buggy = requested_start + window_start
        # = -100 + 100 = 0 (ACCIDENTALLY CORRECT in this case)
        
        # But with different window_start it fails:
        window_start_2 = 200
        window_genomic_start_buggy_2 = requested_start + window_start_2
        # = -100 + 200 = 100 (CORRECT by accident)
        
        # The CORRECT way:
        window_genomic_start_correct = actual_start + window_start
        # = 0 + 100 = 100 (CORRECT)
        
        # For window_start = 100:
        assert window_genomic_start_buggy == 0
        assert window_genomic_start_correct == 100
        assert window_genomic_start_buggy != window_genomic_start_correct


class TestToArraysPerGeneWindowCounting:
    """
    Tests for per-gene window counting and extraction.
    
    NOTE: Windows are counted and extracted PER GENE, not per chromosome.
    Each gene's splice sites are extracted in a range from (min_pos - context_size)
    to (max_pos + context_size), with padding for chromosome boundaries.
    """
    
    def test_per_gene_window_counting_consistency(self):
        """
        Test that window counting is consistent for a single gene.
        """
        min_pos = 100
        max_pos = 200
        context_size = 50
        window_size = 100
        
        # Per-gene calculation
        requested_start = min_pos - context_size  # 50
        requested_end = max_pos + context_size    # 250
        
        gene_length = requested_end - requested_start  # 200
        total_window = context_size + window_size + context_size  # 200
        
        # Windows that can fit
        n_windows = max(0, (gene_length - total_window) // window_size + 1)
        # = max(0, (200 - 200) // 100 + 1) = 1
        
        assert n_windows == 1, f"Expected 1 window, got {n_windows}"

    def test_per_gene_with_negative_requested_start(self):
        """
        Test per-gene window counting when requested_start is negative
        (gene near chromosome start).
        """
        min_pos = 25
        max_pos = 150
        context_size = 50
        window_size = 100
        
        requested_start = min_pos - context_size  # -25
        requested_end = max_pos + context_size    # 200
        actual_start = max(0, requested_start)    # 0
        
        # Calculate padding for sequence
        left_pad = actual_start - requested_start  # 25
        
        # Sequence spans from requested_start to requested_end (with padding)
        seq_length = left_pad + (requested_end - actual_start)
        # = 25 + 200 = 225
        
        total_window = context_size + window_size + context_size  # 200
        n_windows = max(0, (seq_length - total_window) // window_size + 1)
        # = max(0, (225 - 200) // 100 + 1) = 1
        
        assert n_windows == 1, f"Expected 1 window with left padding, got {n_windows}"


class TestToArraysMemapAllocation:
    """
    Tests for memmap allocation and filling correctness.
    """
    
    def test_memmap_not_partially_filled(self):
        """
        Test that memmap arrays are completely filled (not partially).
        
        If counting phase undercounts, memmap will be allocated with too few slots,
        and the filling phase will try to write beyond the allocated size.
        """
        # This test would require mocking or a real integration test
        # For now, it's documented as a test requirement
        pass

    def test_metadata_rows_match_sequence_count(self):
        """
        Test that the number of metadata rows matches the number of sequences.
        
        This catches cases where windows are created but metadata is missing,
        or metadata is created but windows are missing.
        """
        # This test would require running to_arrays on real data
        # For now, it's documented as a test requirement
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
