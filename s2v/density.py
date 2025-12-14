import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

def density_bins_and_error(
    Z: np.ndarray,
    error: np.ndarray,
    num_bins: int = 20,
    bandwidth: float = 0.35,
    max_points: int = 20000,
    seed: int = 0,
):
    """
    Returns: bin_centers, err_prob, counts, low_thr, mid_thr
    where x-axis is log-density (KDE on standardized latent Z).
    """
    rng = np.random.default_rng(seed)
    N_full = Z.shape[0]

    if N_full > max_points:
        idx = rng.choice(N_full, max_points, replace=False)
        Z_use = Z[idx]
        err_use = error[idx]
    else:
        Z_use = Z
        err_use = error

    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z_use)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(Zs)

    log_density = kde.score_samples(Zs)  # log p(z)
    # Use log-density directly (more numerically stable than exp)
    x = log_density

    # Quantile bins in x => equal counts per bin (except ties)
    q = np.linspace(0, 1, num_bins + 1)
    edges = np.quantile(x, q)

    bin_centers = []
    err_prob = []
    counts = []

    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < num_bins - 1:
            mask = (x >= lo) & (x < hi)
        else:
            mask = (x >= lo) & (x <= hi)

        idx_bin = np.where(mask)[0]
        c = len(idx_bin)
        counts.append(c)

        if c == 0:
            err_prob.append(np.nan)
            bin_centers.append(0.5 * (lo + hi))
        else:
            err_prob.append(float(err_use[idx_bin].mean()))
            bin_centers.append(float(0.5 * (lo + hi)))

    # thresholds for shaded regions (based on x-axis distribution)
    low_thr = float(np.quantile(x, 0.33))
    mid_thr = float(np.quantile(x, 0.66))

    return (
        np.array(bin_centers, dtype=np.float64),
        np.array(err_prob, dtype=np.float64),
        counts,
        low_thr,
        mid_thr,
    )
