"""Data splitting utilities for splice site model training."""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Iterable
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import warnings
import os
import gc
import psutil

from .data_loader import MultiGenomeDataLoader
from ..io.splice_sites import SpliceSite


@dataclass
class DataSplit:
    """Container for train/validation/test data splits."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    usage_arrays_train: Optional[Dict[str, np.ndarray]]
    usage_arrays_val: Optional[Dict[str, np.ndarray]]
    usage_arrays_test: Optional[Dict[str, np.ndarray]]
    train_metadata: pd.DataFrame
    val_metadata: pd.DataFrame
    test_metadata: pd.DataFrame
    split_info: Dict


def effective_workers(requested: int, sequential: bool, by_genome: bool) -> int:
    """
    Clamp worker count in memory-saving modes to avoid OOM.
    Returns 1–2 when sequential or by_genome are enabled; otherwise requested.
    """
    if sequential or by_genome:
        return max(1, min(2, requested))
    return max(1, requested)


def get_memory_usage() -> Tuple[float, float]:
    """
    Get current and peak memory usage in MB.
    
    Returns:
        Tuple of (current_mb, peak_mb) in MB
    """
    try:
        process = psutil.Process(os.getpid())
        current_mb = process.memory_info().rss / 1024 / 1024
        # Try to get peak if available (Linux only)
        try:
            with open(f'/proc/{os.getpid()}/status', 'r') as f:
                for line in f:
                    if line.startswith('VmPeak'):
                        peak_mb = int(line.split()[1])
                        return current_mb, peak_mb
        except:
            pass
        return current_mb, current_mb
    except:
        return 0.0, 0.0


def format_memory(mb: float) -> str:
    """
    Format memory value for display.
    
    Args:
        mb: Memory in MB
        
    Returns:
        Formatted string (GB if >= 1024 MB, else MB)
    """
    if mb >= 1024:
        return f"{mb / 1024:.2f}GB"
    else:
        return f"{mb:.0f}MB"


def split_to_memmap_chunked(
    loader: MultiGenomeDataLoader,
    split_sites: Iterable[SpliceSite],
    usage_conditions: List[str],
    mmap_dir: str,
    window_size: int,
    context_size: int,
    alpha_threshold: int,
    requested_workers: int,
    chunk_size: int,
    log_fn=lambda msg: None,
) -> Dict[str, object]:
    """
    Create memmap arrays for sequences, labels, species_ids and usage arrays
    by processing one genome at a time and chunking genes to keep peak memory low.

    Side-effects:
      - Writes arrays under mmap_dir: sequences.npy, labels.npy, species_ids.npy, usage_{i}.npy

    Returns (lightweight):
      {
        "mmap_dir": str,
        "metadata": pd.DataFrame,
        "meta_file": str
      }
    """
    os.makedirs(mmap_dir, exist_ok=True)

    max_memory_mb = 0.0
    chunk_count = 0

    # Group sites by genome
    sites_by_genome: Dict[str, List[SpliceSite]] = {}
    for site in split_sites:
        sites_by_genome.setdefault(site.genome_id, []).append(site)

    log_fn(f"    Found {len(sites_by_genome)} genomes to process")
    for gid, sites in sites_by_genome.items():
        log_fn(f"      {gid}: {len(sites)} sites")

    # Precompute per-genome gene lists
    genes_by_genome: Dict[str, set] = {}
    for site in split_sites:
        genes_by_genome.setdefault(site.genome_id, set()).add(site.gene_id)

    total_windows = 0
    first_write_done = False
    metadata_file = os.path.join(mmap_dir, 'metadata_incremental.csv')
    meta_info: Dict[str, object] = {
        "paths": {},
        "dtypes": {},
        "shapes": {},
        "usage": []
    }

    for genome_idx, (genome_id, genome_sites) in enumerate(sites_by_genome.items(), 1):
        log_fn(f"\n    [{genome_idx}/{len(sites_by_genome)}] Processing {genome_id} ({len(genome_sites)} sites)...")

        genome_genes = sorted(list(genes_by_genome.get(genome_id, set())))
        csize = max(250, chunk_size)
        n_chunks = (len(genome_genes) + csize - 1) // csize
        log_fn(f"        Chunking {len(genome_genes)} genes into {n_chunks} chunks of size ~{csize}")

        sites_by_gene: Dict[str, List[SpliceSite]] = {}
        for s in genome_sites:
            sites_by_gene.setdefault(s.gene_id, []).append(s)

        for ci in range(n_chunks):
            cstart = ci * csize
            cend = min((ci + 1) * csize, len(genome_genes))
            chunk_genes = set(genome_genes[cstart:cend])
            chunk_sites: List[SpliceSite] = []
            for gid in chunk_genes:
                chunk_sites.extend(sites_by_gene.get(gid, []))

            log_fn(f"      Chunk {ci+1}/{n_chunks}: {len(chunk_genes)} genes, {len(chunk_sites)} sites")

            # Process this chunk only
            loader.loaded_data = chunk_sites

            workers = effective_workers(requested_workers, sequential=True, by_genome=True)
            try:
                sequences, labels, usage_arrays, metadata, species_ids = loader.to_arrays(
                    window_size=window_size,
                    context_size=context_size,
                    alpha_threshold=alpha_threshold,
                    n_workers=workers,
                    save_memmap=None
                )
            except MemoryError:
                log_fn("        MemoryError detected, retrying chunk with single worker...")
                sequences, labels, usage_arrays, metadata, species_ids = loader.to_arrays(
                    window_size=window_size,
                    context_size=context_size,
                    alpha_threshold=alpha_threshold,
                    n_workers=1,
                    save_memmap=None
                )

            n_windows = len(sequences)

            if not first_write_done:
                # Initialize memmap files on first write
                seq_shape = (n_windows,) + sequences.shape[1:]
                label_shape = (n_windows,) + labels.shape[1:]
                species_shape = (n_windows,)

                seq_path = os.path.join(mmap_dir, 'sequences.mmap')
                lbl_path = os.path.join(mmap_dir, 'labels.mmap')
                spc_path = os.path.join(mmap_dir, 'species_ids.mmap')

                seq_mm = np.memmap(seq_path, mode='w+', dtype=sequences.dtype, shape=seq_shape)
                seq_mm[:] = sequences; del seq_mm
                lbl_mm = np.memmap(lbl_path, mode='w+', dtype=labels.dtype, shape=label_shape)
                lbl_mm[:] = labels; del lbl_mm
                spc_mm = np.memmap(spc_path, mode='w+', dtype=species_ids.dtype, shape=species_shape)
                spc_mm[:] = species_ids; del spc_mm

                meta_info["paths"].update({"sequences": seq_path, "labels": lbl_path, "species_ids": spc_path})
                meta_info["dtypes"].update({
                    "sequences": str(sequences.dtype),
                    "labels": str(labels.dtype),
                    "species_ids": str(species_ids.dtype),
                })
                meta_info["shapes"].update({
                    "sequences": list(seq_shape),
                    "labels": list(label_shape),
                    "species_ids": list(species_shape),
                })

                # usage arrays
                for key, usage_array in usage_arrays.items():
                    u_shape = (n_windows,) + usage_array.shape[1:]
                    u_path = os.path.join(mmap_dir, f'usage_{key}.mmap')
                    u_mm = np.memmap(u_path, mode='w+', dtype=usage_array.dtype, shape=u_shape)
                    u_mm[:] = usage_array; del u_mm
                    meta_info["usage"].append({"key": key, "path": u_path, "dtype": str(usage_array.dtype), "shape": list(u_shape)})

                total_windows = n_windows
                first_write_done = True
                
                # Write metadata CSV header
                metadata.to_csv(metadata_file, index=False, mode='w')
            else:
                # Append to existing memmap files more efficiently
                new_total = total_windows + n_windows

                for array_name, array_file in [
                    ('sequences', meta_info["paths"]["sequences"]),
                    ('labels', meta_info["paths"]["labels"]),
                    ('species_ids', meta_info["paths"]["species_ids"])
                ]:
                    if array_name == 'sequences':
                        current_array = sequences
                    elif array_name == 'labels':
                        current_array = labels
                    else:
                        current_array = species_ids

                    old_dtype = np.dtype(meta_info["dtypes"][array_name])
                    old_shape_tail = tuple(meta_info["shapes"][array_name][1:])
                    new_shape = (new_total,) + old_shape_tail
                    
                    old_mm = np.memmap(array_file, mode='r+', dtype=old_dtype, shape=(total_windows,) + old_shape_tail)
                    tmp_path = os.path.join(mmap_dir, f'{array_name}_temp.mmap')
                    new_mm = np.memmap(tmp_path, mode='w+', dtype=old_dtype, shape=new_shape)
                    new_mm[:total_windows] = old_mm[:]
                    new_mm[total_windows:new_total] = current_array
                    del old_mm, new_mm
                    os.replace(tmp_path, array_file)
                    meta_info["shapes"][array_name][0] = new_total

                # Usage arrays
                for key, usage_array in usage_arrays.items():
                    entry = next((e for e in meta_info["usage"] if e["key"] == key), None)
                    if entry is None:
                        u_path = os.path.join(mmap_dir, f'usage_{key}.mmap')
                        u_shape_init = (total_windows,) + usage_array.shape[1:]
                        u_mm = np.memmap(u_path, mode='w+', dtype=usage_array.dtype, shape=u_shape_init)
                        u_mm[:] = 0; del u_mm
                        entry = {"key": key, "path": u_path, "dtype": str(usage_array.dtype), "shape": list(u_shape_init)}
                        meta_info["usage"].append(entry)
                    
                    u_path = entry["path"]
                    u_dtype = np.dtype(entry["dtype"])
                    u_tail = tuple(entry["shape"][1:])
                    u_new_shape = (new_total,) + u_tail
                    u_old = np.memmap(u_path, mode='r+', dtype=u_dtype, shape=(total_windows,) + u_tail)
                    u_tmp = os.path.join(mmap_dir, f'usage_{key}_temp.mmap')
                    u_new = np.memmap(u_tmp, mode='w+', dtype=u_dtype, shape=u_new_shape)
                    u_new[:total_windows] = u_old[:]
                    u_new[total_windows:new_total] = usage_array
                    del u_old, u_new
                    os.replace(u_tmp, u_path)
                    entry["shape"][0] = new_total

                total_windows = new_total
                
                # Append metadata CSV
                metadata.to_csv(metadata_file, index=False, mode='a', header=False)

            # Clean up immediately after writing
            del sequences, labels, usage_arrays, species_ids, metadata
            gc.collect()

            # Report memory usage occasionally
            chunk_count += 1
            current_mem, peak_mem = get_memory_usage()
            max_memory_mb = max(max_memory_mb, current_mem)
            log_fn(f"        Memory: current={format_memory(current_mem)}, peak in session={format_memory(max_memory_mb)}")

        log_fn(f"        {genome_id} processed: {total_windows} total windows")

    # Load final metadata from CSV
    combined_metadata = pd.read_csv(metadata_file)

    # Save metadata.json
    meta_out = {
        "paths": meta_info["paths"],
        "dtypes": meta_info["dtypes"],
        "shapes": meta_info["shapes"],
        "usage": meta_info["usage"],
        "total_windows": meta_info["shapes"]["sequences"][0] if meta_info["shapes"].get("sequences") else 0
    }
    import json
    with open(os.path.join(mmap_dir, "metadata.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    # Clean up incremental CSV
    if os.path.exists(metadata_file):
        os.remove(metadata_file)

    # Final memory report
    final_mem, peak_mem = get_memory_usage()
    log_fn(f"\n    Final memory usage: current={format_memory(final_mem)}, peak={format_memory(max_memory_mb)}")

    return {
        "mmap_dir": mmap_dir,
        "metadata": combined_metadata,
        "meta_file": os.path.join(mmap_dir, "metadata.json"),
    }
