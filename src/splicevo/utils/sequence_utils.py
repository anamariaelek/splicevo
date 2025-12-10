"""Sequence utility functions for DNA processing."""

import numpy as np

def one_hot_encode(seq: str):
    """Encode a DNA sequence into one-hot representation"""
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    ohe = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nuc in enumerate(seq):
        idx = nuc_to_idx.get(nuc, 4)
        if idx < 4:
            ohe[i, idx] = 1.0
    return ohe

def one_hot_decode(ohe_seq):
    """Decode one-hot encoded sequence to string"""
    bases = ['A', 'C', 'G', 'T']
    if isinstance(ohe_seq, np.ndarray):
        return ''.join(bases[np.argmax(ohe_seq[i])] for i in range(len(ohe_seq)))
    else:
        return ''.join(bases[np.argmax(ohe_seq[i].numpy())] for i in range(len(ohe_seq)))


def complement_sequence(seq: str) -> str:
    """
    Generate the complement of a DNA sequence (not reverse complement).
    
    For negative strand genes, we keep the sequence reversed but apply
    nucleotide complementation: A<->T, C<->G.
    
    Args:
        seq: DNA sequence string
        
    Returns:
        Complemented sequence
    """
    comp_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp_map.get(nuc.upper(), 'N') for nuc in seq)
