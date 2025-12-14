#!/usr/bin/env python3
import argparse
import numpy as np

from s2v.io import load_inputs, save_outputs_dir
from s2v.embedders.from_npy import NpyEmbedder
from s2v.density import density_bins_and_error
from s2v.plots import (
    plot_umap_with_density_and_errors,
    plot_density_error_curve,
)

def parse_args():
    p = argparse.ArgumentParser(description="Syndrome2Vec-style decoder analysis (model-optional).")
    p.add_argument("--syndromes", type=str, required=True, help="Path to syndromes.npy (N x D or N x T x D).")
    p.add_argument("--decoder_outputs", type=str, required=True, help="Path to decoder_outputs.npy (N,) or (N,1).")
    p.add_argument("--logical_labels", type=str, required=True, help="Path to logical_labels.npy (N,) or (N,1).")

    # If embeddings are already computed, no model needed
    p.add_argument("--embeddings", type=str, default=None, help="Optional path to embeddings.npy (N x k).")

    # UMAP + density settings
    p.add_argument("--umap_neighbors", type=int, default=30)
    p.add_argument("--umap_min_dist", type=float, default=0.05)
    p.add_argument("--umap_seed", type=int, default=0)

    p.add_argument("--kde_bandwidth", type=float, default=0.35)
    p.add_argument("--kde_max_points", type=int, default=20000)

    p.add_argument("--num_bins", type=int, default=20)
    p.add_argument("--outdir", type=str, default="outputs")

    return p.parse_args()

def main():
    args = parse_args()
    synd, dec_out, labels = load_inputs(args.syndromes, args.decoder_outputs, args.logical_labels)

    # MWPM (or decoder) error indicator
    # dec_out and labels are (N,) uint8/0-1
    mwpm_error = (dec_out != labels).astype(np.uint8)

    if args.embeddings is None:
        raise SystemExit(
            "No --embeddings provided. This repo is analysis-only by default.\n"
            "Either provide embeddings.npy OR add an embedder plugin (e.g. your private model) and call it here."
        )

    Z = np.load(args.embeddings)
    if Z.ndim != 2 or Z.shape[0] != synd.shape[0]:
        raise ValueError(f"embeddings shape {Z.shape} incompatible with N={synd.shape[0]}")

    outdir = save_outputs_dir(args.outdir)

    # 1) UMAP 2D projection (for visualization only)
    # (we compute it inside plot function to keep this main file clean)

    # 2) Density-vs-error curve in latent space (KDE on standardized Z)
    curve = density_bins_and_error(
        Z=Z,
        error=mwpm_error,
        num_bins=args.num_bins,
        bandwidth=args.kde_bandwidth,
        max_points=args.kde_max_points,
        seed=args.umap_seed,
    )
    bin_centers, err_prob, counts, low_thr, mid_thr = curve

    # 3) Plots
    plot_umap_with_density_and_errors(
        Z=Z,
        error=mwpm_error,
        outpath=f"{outdir}/umap_density_errors.png",
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.umap_seed,
    )

    plot_density_error_curve(
        bin_centers=bin_centers,
        err_prob=err_prob,
        low_thr=low_thr,
        mid_thr=mid_thr,
        outpath=f"{outdir}/density_vs_error.png",
        xlabel="Latent log-density (KDE; standardized latent space)",
        title="Decoder logical error probability vs latent density",
    )

    # 4) Save coordinates + curve data for paper reproducibility
    np.save(f"{outdir}/mwpm_error.npy", mwpm_error)
    np.save(f"{outdir}/latent_Z.npy", Z)
    np.save(f"{outdir}/curve_bin_centers.npy", bin_centers)
    np.save(f"{outdir}/curve_err_prob.npy", err_prob)
    np.save(f"{outdir}/curve_counts.npy", np.array(counts, dtype=np.int64))

    with open(f"{outdir}/curve_points.csv", "w") as f:
        f.write("bin_center,err_prob,count\n")
        for x, y, c in zip(bin_centers, err_prob, counts):
            y_str = "nan" if (y is None or (isinstance(y, float) and np.isnan(y))) else f"{y:.8f}"
            f.write(f"{x:.8f},{y_str},{c}\n")

    print("\nSaved outputs to:", outdir)
    print("Counts per bin:", counts)
    print("Low threshold:", low_thr, "Mid threshold:", mid_thr)

if __name__ == "__main__":
    main()
