import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import umap

def plot_umap_with_density_and_errors(
    Z: np.ndarray,
    error: np.ndarray,
    outpath: str,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    seed: int = 0,
):
    """
    Produces a publication-friendly UMAP plot:
      - background: KDE density in UMAP space (heatmap)
      - overlay: error points (red) vs non-error (black)
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        metric="euclidean",
    )
    U = reducer.fit_transform(Z)

    # KDE in 2D UMAP space for background
    scaler = StandardScaler()
    Us = scaler.fit_transform(U)

    kde = KernelDensity(kernel="gaussian", bandwidth=0.35)
    kde.fit(Us)

    # Create grid
    x_min, x_max = Us[:, 0].min(), Us[:, 0].max()
    y_min, y_max = Us[:, 1].min(), Us[:, 1].max()
    pad_x = 0.08 * (x_max - x_min + 1e-9)
    pad_y = 0.08 * (y_max - y_min + 1e-9)
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y

    grid_n = 250
    xs = np.linspace(x_min, x_max, grid_n)
    ys = np.linspace(y_min, y_max, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.c_[XX.ravel(), YY.ravel()]

    log_d = kde.score_samples(grid).reshape(grid_n, grid_n)

    # Plot
    plt.figure(figsize=(9.5, 7.0))
    plt.imshow(
        log_d,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto",
    )

    # Overlay points
    err = (error.astype(bool))
    plt.scatter(Us[~err, 0], Us[~err, 1], s=6, alpha=0.25)
    plt.scatter(Us[err, 0],  Us[err, 1],  s=10, alpha=0.9)

    plt.title("UMAP of latent embeddings with decoder failures overlaid")
    plt.xlabel("UMAP-1 (standardized)")
    plt.ylabel("UMAP-2 (standardized)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()


def plot_density_error_curve(
    bin_centers: np.ndarray,
    err_prob: np.ndarray,
    low_thr: float,
    mid_thr: float,
    outpath: str,
    xlabel: str,
    title: str,
):
    # Clean NaNs for plotting: draw line through finite points
    finite = np.isfinite(err_prob)

    plt.figure(figsize=(10, 6))
    xmin = float(np.min(bin_centers))
    xmax = float(np.max(bin_centers))

    # Shaded regions
    plt.axvspan(xmin, low_thr,  alpha=0.22, label="Low-density (rare)")
    plt.axvspan(low_thr, mid_thr, alpha=0.22, label="Medium-density")
    plt.axvspan(mid_thr, xmax,  alpha=0.22, label="High-density (common)")

    plt.axvline(low_thr, linestyle="--", linewidth=1.2, alpha=0.8)
    plt.axvline(mid_thr, linestyle="--", linewidth=1.2, alpha=0.8)

    # Plot (line through finite)
    plt.plot(bin_centers[finite], err_prob[finite], "-o", linewidth=2.2, markersize=7)

    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel("Logical error probability", fontsize=13)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()
