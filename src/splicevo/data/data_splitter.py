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
        # Reduce chunk size more aggressively to prevent OOM during conversion
        csize = max(100, min(500, chunk_size))
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
            
            # Pre-check memory before attempting array conversion
            mem_before, _ = get_memory_usage()
            max_memory_mb = max(max_memory_mb, mem_before)
            estimated_chunk_mb = (len(chunk_sites) * 1900 * 4 * 4) / (1024 * 1024)  # rough estimate
            if mem_before + estimated_chunk_mb > 3500:
                workers = 1
                log_fn(f"        Memory approaching limit ({format_memory(mem_before)}), forcing single worker")
            
            try:
                sequences, labels, usage_arrays, metadata, species_ids = loader.to_arrays(
                    window_size=window_size,
                    context_size=context_size,
                    alpha_threshold=alpha_threshold,
                    n_workers=workers,
                    save_memmap=None
                )
            except MemoryError as e:
                log_fn(f"        MemoryError: {e}")
                log_fn("        Retrying chunk with single worker and smaller operations...")
                try:
                    sequences, labels, usage_arrays, metadata, species_ids = loader.to_arrays(
                        window_size=window_size,
                        context_size=context_size,
                        alpha_threshold=alpha_threshold,
                        n_workers=1,
                        save_memmap=None
                    )
                except MemoryError:
                    log_fn(f"        FATAL: Still OOM after retry. Chunk {ci+1}/{n_chunks} is too large.")
                    log_fn(f"        Try reducing --chunk-size from {chunk_size} to {max(50, chunk_size // 2)}")
                    raise

            n_windows = len(sequences)

            if not first_write_done:
                # Initialize memmap files on first write
                seq_shape = (n_windows,) + sequences.shape[1:]
                label_shape = (n_windows,) + labels.shape[1:]
                species_shape = (n_windows,)

                seq_path = os.path.join(mmap_dir, 'sequences.mmap')
                lbl_path = os.path.join(mmap_dir, 'labels.mmap')
                spc_path = os.path.join(mmap_dir, 'species_ids.mmap')

                # Write sequences
                seq_mm = np.memmap(seq_path, mode='w+', dtype=sequences.dtype, shape=seq_shape)
                seq_mm[:] = sequences
                del seq_mm
                del sequences
                
                # Write labels
                lbl_mm = np.memmap(lbl_path, mode='w+', dtype=labels.dtype, shape=label_shape)
                lbl_mm[:] = labels
                del lbl_mm
                del labels
                
                # Write species_ids
                spc_mm = np.memmap(spc_path, mode='w+', dtype=species_ids.dtype, shape=species_shape)
                spc_mm[:] = species_ids
                del spc_mm
                del species_ids
                gc.collect()

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

                # usage arrays - write directly
                for key, usage_array in usage_arrays.items():
                    u_shape = (n_windows,) + usage_array.shape[1:]
                    u_path = os.path.join(mmap_dir, f'usage_{key}.mmap')
                    u_mm = np.memmap(u_path, mode='w+', dtype=usage_array.dtype, shape=u_shape)
                    u_mm[:] = usage_array
                    del u_mm
                    del usage_array
                    meta_info["usage"].append({"key": key, "path": u_path, "dtype": str(usage_array.dtype), "shape": list(u_shape)})
                
                del usage_arrays
                gc.collect()

                total_windows = n_windows
                first_write_done = True
                
                # Write metadata CSV header
                metadata.to_csv(metadata_file, index=False, mode='w')
                del metadata
                
            else:
                # Append mode: extend memmap files without loading all old data
                new_total = total_windows + n_windows
                old_size = total_windows

                # Extend and write sequences (no copy of old data)
                seq_path = meta_info["paths"]["sequences"]
                old_seq = np.memmap(seq_path, mode='r+', dtype=meta_info["dtypes"]["sequences"], 
                                   shape=(old_size,) + sequences.shape[1:])
                # Resize by recreating memmap with new shape
                del old_seq
                os.system(f"truncate -s +{n_windows * np.dtype(meta_info['dtypes']['sequences']).itemsize * sequences.shape[1] * sequences.shape[2]} {seq_path}")
                seq_mm = np.memmap(seq_path, mode='r+', dtype=meta_info["dtypes"]["sequences"],
                                  shape=(new_total,) + sequences.shape[1:])
                seq_mm[old_size:new_total] = sequences
                del seq_mm
                del sequences
                gc.collect()

                # Extend and write labels
                lbl_path = meta_info["paths"]["labels"]
                lbl_mm = np.memmap(lbl_path, mode='r+', dtype=meta_info["dtypes"]["labels"],
                                  shape=(new_total,) + labels.shape[1:])
                lbl_mm[old_size:new_total] = labels
                del lbl_mm
                del labels
                gc.collect()

                # Extend and write species_ids
                spc_path = meta_info["paths"]["species_ids"]
                spc_mm = np.memmap(spc_path, mode='r+', dtype=meta_info["dtypes"]["species_ids"],
                                  shape=(new_total,))
                spc_mm[old_size:new_total] = species_ids
                del spc_mm
                del species_ids
                gc.collect()

                # Extend and write usage arrays
                for key, usage_array in usage_arrays.items():
                    entry = next((e for e in meta_info["usage"] if e["key"] == key), None)
                    if entry is None:
                        # First time seeing this usage key
                        u_path = os.path.join(mmap_dir, f'usage_{key}.mmap')
                        # Need to backfill with placeholder
                        placeholder = np.zeros((old_size,) + usage_array.shape[1:], dtype=usage_array.dtype)
                        u_mm = np.memmap(u_path, mode='w+', dtype=usage_array.dtype, 
                                       shape=(new_total,) + usage_array.shape[1:])
                        u_mm[:old_size] = placeholder
                        u_mm[old_size:new_total] = usage_array
                        del u_mm, placeholder
                        entry = {"key": key, "path": u_path, "dtype": str(usage_array.dtype), "shape": list((new_total,) + usage_array.shape[1:])}
                        meta_info["usage"].append(entry)
                    else:
                        u_path = entry["path"]
                        u_mm = np.memmap(u_path, mode='r+', dtype=entry["dtype"],
                                       shape=(new_total,) + usage_array.shape[1:])
                        u_mm[old_size:new_total] = usage_array
                        del u_mm
                        entry["shape"][0] = new_total
                    
                    del usage_array
                
                del usage_arrays
                gc.collect()

                total_windows = new_total
                
                # Append metadata CSV
                metadata.to_csv(metadata_file, index=False, mode='a', header=False)
                del metadata

            # Convert to memmap and explicitly release numpy arrays
            # Write sequences
            seq_mm = np.memmap(meta_info["paths"]["sequences"], mode='r+', 
                             dtype=sequences.dtype, shape=(total_windows,) + sequences.shape[1:])
            seq_mm[total_windows - n_windows:total_windows] = sequences
            del seq_mm
            del sequences
            gc.collect()
            
            # Write labels
            lbl_mm = np.memmap(meta_info["paths"]["labels"], mode='r+',
                             dtype=labels.dtype, shape=(total_windows,) + labels.shape[1:])
            lbl_mm[total_windows - n_windows:total_windows] = labels
            del lbl_mm
            del labels
            gc.collect()
            
            # Write species_ids
            spc_mm = np.memmap(meta_info["paths"]["species_ids"], mode='r+',
                             dtype=species_ids.dtype, shape=(total_windows,))
            spc_mm[total_windows - n_windows:total_windows] = species_ids
            del spc_mm
            del species_ids
            gc.collect()
            
            # Write usage arrays
            for key, usage_array in usage_arrays.items():
                entry = next((e for e in meta_info["usage"] if e["key"] == key), None)
                if entry is not None:
                    u_mm = np.memmap(entry["path"], mode='r+', 
                                   dtype=usage_array.dtype, shape=(total_windows,) + usage_array.shape[1:])
                    u_mm[total_windows - n_windows:total_windows] = usage_array
                    del u_mm
                del usage_array
            
            del usage_arrays
            gc.collect()

            # Report memory usage with delta
            chunk_count += 1
            current_mem, peak_mem = get_memory_usage()
            max_memory_mb = max(max_memory_mb, current_mem)
            mem_delta = current_mem - mem_before
            log_fn(f"        Memory: current={format_memory(current_mem)}, delta={format_memory(mem_delta)}, peak={format_memory(max_memory_mb)}")

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
