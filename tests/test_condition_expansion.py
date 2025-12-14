"""
Test script to verify the save_attributions_for_modisco condition expansion feature
and its integration with AttributionAggregator.aggregate_by_condition.
"""

import numpy as np
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from splicevo.attributions.compute import save_attributions_for_modisco
from splicevo.attributions.modisco_analysis import AttributionAggregator


def test_condition_expansion():
    """Test that save_attributions_for_modisco correctly expands multi-condition attributions."""
    
    print("=" * 70)
    print("Test 1: Condition Expansion in save_attributions_for_modisco")
    print("=" * 70)
    
    # Create test data with 3D attributions (multiple conditions)
    # Sequence length must be > context + position + window
    seq_len = 500  # Large enough for window=100, position=100, context=50
    attrs_dict = {
        'site_0_pos_100': {
            'sequence': np.random.randn(seq_len, 4),
            'attribution': np.random.randn(seq_len, 4, 3),  # 3D: 3 conditions
            'position': 100,
            'context': 50,
            'metadata': {
                'genome_id': 'test_genome',
                'chromosome': '1',
                'genomic_coord': '1000',
                'window_start': '950',
                'window_end': '1049',
                'strand': '+',
            },
            'site_type': 'donor',
            'site_class': 1,
        },
        'site_1_pos_95': {
            'sequence': np.random.randn(seq_len, 4),
            'attribution': np.random.randn(seq_len, 4, 3),  # 3D: 3 conditions
            'position': 95,
            'context': 50,
            'metadata': {
                'genome_id': 'test_genome',
                'chromosome': '1',
                'genomic_coord': '2000',
                'window_start': '1950',
                'window_end': '2049',
                'strand': '-',
            },
            'site_type': 'acceptor',
            'site_class': 2,
        }
    }
    
    condition_names = ['alpha', 'beta', 'sse']
    
    # Test 1a: Expand without specific condition_idx
    print("\nTest 1a: Expand all conditions (condition_idx=None)")
    result = save_attributions_for_modisco(
        attrs_dict,
        output_path='/tmp/test_expand_all',
        window=100,
        condition_idx=None,  # Expand all
        condition_names=condition_names,
        verbose=False
    )
    
    print(f"  ✓ Input sites: 2")
    print(f"  ✓ Conditions: {result['n_conditions']}")
    print(f"  ✓ Output sites: {result['n_sites']} (expected 2 * 3 = 6)")
    
    # Check metadata for condition tracking
    assert result['n_sites'] == 6, f"Expected 6 expanded sites, got {result['n_sites']}"
    
    # Load metadata to verify condition tracking
    with open(result['metadata_path'], 'r') as f:
        metadata = json.load(f)
    
    print(f"\n  Metadata verification:")
    print(f"    Total sites: {metadata['n_sites']}")
    print(f"    n_conditions: {metadata['n_conditions']}")
    print(f"    condition_names: {metadata['condition_names']}")
    
    # Count conditions in metadata
    condition_counts = {}
    for site in metadata['sites']:
        cond_name = site.get('condition_name', 'none')
        condition_counts[cond_name] = condition_counts.get(cond_name, 0) + 1
    
    print(f"\n  Sites per condition:")
    for cond, count in sorted(condition_counts.items()):
        print(f"    {cond}: {count}")
    
    assert all(count == 2 for count in condition_counts.values()), \
        "Each condition should have 2 sites"
    
    print("\n  ✓ Test 1a PASSED: Condition expansion works correctly")
    
    # Test 1b: Extract specific condition
    print("\nTest 1b: Extract specific condition (condition_idx=1)")
    result_single = save_attributions_for_modisco(
        attrs_dict,
        output_path='/tmp/test_single_condition',
        window=100,
        condition_idx=1,  # Only extract 'beta'
        condition_names=condition_names,
        verbose=False
    )
    
    print(f"  ✓ Output sites: {result_single['n_sites']} (expected 2)")
    assert result_single['n_sites'] == 2, f"Expected 2 sites, got {result_single['n_sites']}"
    print("  ✓ Test 1b PASSED: Single condition extraction works")
    
    return result, result_single


def test_aggregator_integration(expanded_result):
    """Test that AttributionAggregator.aggregate_by_condition works with expanded data."""
    
    print("\n" + "=" * 70)
    print("Test 2: AttributionAggregator.aggregate_by_condition Integration")
    print("=" * 70)
    
    # Load the expanded data
    sequences = np.load(expanded_result['sequences_path'])
    attributions = np.load(expanded_result['attributions_path'])
    
    with open(expanded_result['metadata_path'], 'r') as f:
        metadata = json.load(f)
    
    print(f"\nLoaded data:")
    print(f"  Sequences shape: {sequences.shape}")
    print(f"  Attributions shape: {attributions.shape}")
    
    # Create 4D attributions for aggregator test
    # Reshape (n_sites*n_conditions, seq_len, 4) back to (n_sites, seq_len, 4, n_conditions)
    n_sites_expanded = sequences.shape[0]
    n_conditions = metadata['n_conditions']
    seq_len = sequences.shape[1]
    
    # Reshape to 4D by condition
    attributions_4d = np.zeros((n_sites_expanded // n_conditions, seq_len, 4, n_conditions))
    sequences_unique = sequences[::n_conditions]  # One sequence per site
    
    # Redistribute attributions by condition
    for i in range(n_conditions):
        indices = np.arange(i, n_sites_expanded, n_conditions)
        attributions_4d[:, :, :, i] = attributions[indices]
    
    print(f"  Reshaped attributions to 4D: {attributions_4d.shape}")
    
    # Test aggregation
    aggregator = AttributionAggregator(verbose=True)
    condition_names = metadata['condition_names']
    
    print(f"\nAggregating by condition with names: {condition_names}")
    
    groups = aggregator.aggregate_by_condition(
        attributions_4d,
        sequences_unique,
        condition_names,
        metadata={
            'source': 'test',
            'n_sites_expanded': n_sites_expanded,
        }
    )
    
    print(f"\nAggregation result:")
    for group_name, group_input in groups.items():
        print(f"  {group_name}: {group_input.attributions.shape}")
        assert group_input.attributions.shape[0] == len(sequences_unique), \
            f"Each condition group should have {len(sequences_unique)} sites"
    
    print("\n  ✓ Test 2 PASSED: AttributionAggregator works with expanded conditions")
    
    return groups


def test_metadata_preservation():
    """Test that metadata is correctly preserved through expansion."""
    
    print("\n" + "=" * 70)
    print("Test 3: Metadata Preservation Through Expansion")
    print("=" * 70)
    
    attrs_dict = {
        'site_0': {
            'sequence': np.random.randn(500, 4),
            'attribution': np.random.randn(500, 4, 2),
            'position': 100,
            'context': 50,
            'metadata': {
                'genome_id': 'human',
                'chromosome': '3',
                'genomic_coord': '12345',
                'window_start': '12295',
                'window_end': '12394',
                'strand': '+',
            },
            'site_type': 'donor',
            'seq_idx': 42,
        }
    }
    
    result = save_attributions_for_modisco(
        attrs_dict,
        output_path='/tmp/test_metadata',
        window=100,
        condition_idx=None,
        condition_names=['cond_A', 'cond_B'],
        verbose=False
    )
    
    with open(result['metadata_path'], 'r') as f:
        metadata = json.load(f)
    
    print(f"\nMetadata check:")
    
    for i, site in enumerate(metadata['sites']):
        print(f"\n  Site {i}:")
        print(f"    condition_name: {site.get('condition_name')}")
        print(f"    condition_idx: {site.get('condition_idx')}")
        print(f"    site_type: {site.get('site_type')}")
        print(f"    genome_id: {site.get('genome_id')}")
        print(f"    chromosome: {site.get('chromosome')}")
        print(f"    genomic_coord: {site.get('genomic_coord')}")
        print(f"    window_start: {site.get('window_start')}")
        print(f"    window_end: {site.get('window_end')}")
        
        # Verify metadata is preserved
        assert site['site_type'] == 'donor', "site_type should be preserved"
        assert site['genome_id'] == 'human', "genome_id should be preserved"
        assert site['chromosome'] == '3', "chromosome should be preserved"
        assert site['seq_idx'] == 42, "seq_idx should be preserved"
    
    print("\n  ✓ Test 3 PASSED: All metadata correctly preserved")


if __name__ == '__main__':
    try:
        # Run tests
        expanded_result, single_result = test_condition_expansion()
        groups = test_aggregator_integration(expanded_result)
        test_metadata_preservation()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ Condition expansion works correctly")
        print("  ✓ Single condition extraction works")
        print("  ✓ AttributionAggregator.aggregate_by_condition integration works")
        print("  ✓ Metadata preservation through expansion works")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
