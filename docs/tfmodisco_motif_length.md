# TF-MoDISco Motif Length Determination

## Overview

The length of motifs returned by TF-MoDISco is **NOT directly determined by `sliding_window_size` or `flank_size`**, but rather by parameters in the **motif polishing stage** that occur after seqlet clustering.

## Current Settings and Motif Length

With the default configuration in `ModiscoConfig`:
```python
sliding_window_size: int = 12
flank_size: int = 10
```

**The output motif length is 50 bases**, determined by:
$$\text{Final Motif Length} = \text{trim\_to\_window\_size} + 2 \times \text{initial\_flank\_to\_add}$$
$$50 = 30 + 2(10)$$

## TF-MoDISco Algorithm Stages

### 1. **Seqlet Extraction** (Uses sliding_window_size and flank_size)
```
Initial seqlet size = sliding_window_size + 2 * flank_size
                    = 12 + 2 * 10 = 32 bases
```

- **sliding_window_size (12)**: Core window size scanned across sequences
- **flank_size (10)**: Bases added on each side during extraction
- Result: ~32-base seqlets are extracted from high-importance regions

### 2. **Seqlet Clustering**
- Seqlets are grouped into clusters (metaclusters)
- Controlled by parameters:
  - `min_metacluster_size` (20)
  - `max_seqlets_per_metacluster` (100000)
  - `target_seqlet_fdr` (0.05)

### 3. **Motif Polishing** (Determines final motif length)

This is where the output motif length is determined:

**a) Trimming to Information-Rich Core:**
- Finds the `trim_to_window_size` (default: 30 bases) region with highest information content
- Default in modiscolite: **30 bases**
- This is NOT exposed in our current `ModiscoConfig`

**b) Re-expanding with Flanks:**
- Adds `initial_flank_to_add` bases on each side (default: 10 bases)
- Default in modiscolite: **10 bases**
- This is NOT exposed in our current `ModiscoConfig`

**c) Optional Final Expansion:**
- Can add `final_flank_to_add` bases (default: 0)
- Rarely used

**Final Formula:**
```
Final motif length = trim_to_window_size + 2 * initial_flank_to_add
                   = 30 + 2 * 10 = 50 bases
```

## Parameter Meanings

| Parameter | Value | Stage | Purpose |
|-----------|-------|-------|---------|
| `sliding_window_size` | 12 | Extraction | Core window for scanning |
| `flank_size` | 10 | Extraction | Initial flank during seqlet extraction |
| `trim_to_window_size` | 30* | Polishing | Trim to information-rich core |
| `initial_flank_to_add` | 10* | Polishing | Re-expand after trimming |
| `final_flank_to_add` | 0* | Polishing | Final optional expansion |
| `window_size` (save_hdf5) | 400 | Storage | Window around peak (not motif length) |

*These are hardcoded defaults in modiscolite and not exposed in our `ModiscoConfig`

## Controlling Motif Length

To change the output motif length, you would need to:

1. **Modify the modiscolite source code** to expose these parameters:
   - `trim_to_window_size` (currently ~30)
   - `initial_flank_to_add` (currently ~10)

2. **Or pass them directly to `tfmodisco.TFMoDISco()`** if modiscolite supports it

### Example Target Motif Lengths

| Target Length | trim_to_window_size | initial_flank_to_add | Calculation |
|---------------|-------------------|----------------------|-------------|
| 30 | 20 | 5 | 20 + 2(5) |
| 50 | 30 | 10 | 30 + 2(10) ✓ Current |
| 70 | 40 | 15 | 40 + 2(15) |
| 100 | 60 | 20 | 60 + 2(20) |

## Why These Settings?

The current settings (sliding_window_size=12, flank_size=10) are reasonable because:
1. **12 bases** is small enough to capture local binding features
2. **10-base flanks** provide context for regulatory elements
3. **Result: 50 bases** captures typical transcription factor binding sites and regulatory elements
4. **Balance**: Large enough for specificity, small enough for interpretability


## References

- TF-MoDISco paper: "Discovering regulatory DNA motifs with TF-MoDISco"
- Modiscolite GitHub: https://github.com/jbkinney/modiscolite
- The polishing stage is described in the original MoDISco method for seqlet optimization
