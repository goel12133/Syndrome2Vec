import os
import numpy as np

def _load_vec(path: str) -> np.ndarray:
    x = np.load(path)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array from {path}, got shape {x.shape}")
    return x

def load_inputs(syndromes_path: str, decoder_outputs_path: str, logical_labels_path: str):
    synd = np.load(syndromes_path)
    if synd.ndim not in (2, 3):
        raise ValueError(f"syndromes.npy must be (N,D) or (N,T,D); got {synd.shape}")
    N = synd.shape[0]

    dec_out = _load_vec(decoder_outputs_path).astype(np.uint8)
    labels  = _load_vec(logical_labels_path).astype(np.uint8)

    if dec_out.shape[0] != N or labels.shape[0] != N:
        raise ValueError(f"Length mismatch: synd N={N}, decoder={len(dec_out)}, labels={len(labels)}")

    return synd, dec_out, labels

def save_outputs_dir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir
