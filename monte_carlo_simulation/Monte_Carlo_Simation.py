"""
monte_carlo_aggregate.py

Monte Carlo simulation for aggregate insurance claims:
    S = sum_{i=1}^N X_i
where N ~ Poisson(lambda) and X_i ~ severity distribution (lognormal or gamma).

Usage (example):
    python monte_carlo_aggregate.py --n-sim 200000 --lambda 2.5 --severity lognormal \
        --ln-mu 8.5 --ln-sigma 1.2 --out-dir outputs --save-plots

This script:
- Simulates aggregate losses
- Computes mean, std, VaR (quantiles), and TVaR (conditional tail mean)
- Produces plots: histogram (linear), histogram (log x-axis), empirical CDF
- Saves a CSV of simulated aggregates and optional plot PNGs

Requirements:
    numpy, pandas, matplotlib

Author: Evan Whitfield
Date: 2025-10-12
"""

from __future__ import annotations
import argparse
import os
from typing import Tuple, Dict, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Simulation core
# -------------------------
def simulate_aggregate(
    n_sim: int = 100_000,
    freq_lambda: float = 2.0,
    severity_dist: str = "lognormal",  # "lognormal" or "gamma"
    lognormal_mu: float = 8.0,
    lognormal_sigma: float = 1.0,
    gamma_shape: float = 2.0,
    gamma_scale: float = 2000.0,
    random_seed: int | None = 12345,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate aggregate losses S = sum_{i=1}^N X_i.

    Returns:
        aggregates: ndarray, shape (n_sim,) aggregate loss per simulation
        freqs: ndarray, shape (n_sim,) drawn frequencies (N)
    """
    rng = np.random.default_rng(random_seed)

    # Draw frequencies
    N = rng.poisson(lam=freq_lambda, size=n_sim)

    aggregates = np.zeros(n_sim, dtype=float)

    # Vectorized approach by grouping simulations with same count
    unique_counts = np.unique(N)
    if severity_dist.lower() == "lognormal":
        for uc in unique_counts:
            if uc == 0:
                continue
            mask = (N == uc)
            # Draw (num_masked, uc) shape and sum along axis 1
            sev = rng.lognormal(mean=lognormal_mu, sigma=lognormal_sigma, size=(mask.sum(), uc))
            aggregates[mask] = sev.sum(axis=1)
    elif severity_dist.lower() == "gamma":
        for uc in unique_counts:
            if uc == 0:
                continue
            mask = (N == uc)
            sev = rng.gamma(shape=gamma_shape, scale=gamma_scale, size=(mask.sum(), uc))
            aggregates[mask] = sev.sum(axis=1)
    else:
        raise ValueError("severity_dist must be 'lognormal' or 'gamma'")

    return aggregates, N

# -------------------------
# Risk metrics
# -------------------------
def risk_metrics(aggregates: np.ndarray, alphas: Sequence[float] = (0.9, 0.95, 0.99)) -> Dict:
    """
    Compute mean, std, VaR (quantiles), and TVaR for given alpha levels.

    TVaR (Conditional Tail Expectation) at alpha is E[S | S > VaR_alpha]
    """
    aggregates = np.asarray(aggregates)
    mean = float(aggregates.mean())
    std = float(aggregates.std(ddof=0))
    alphas = tuple(alphas)
    quantiles = np.quantile(aggregates, alphas)
    tvars = []
    for q in quantiles:
        tail = aggregates[aggregates > q]
        tvars.append(float(np.nan) if tail.size == 0 else float(tail.mean()))
    return {
        "mean": mean,
        "std": std,
        "alphas": alphas,
        "VaR": quantiles,
        "TVaR": np.array(tvars, dtype=float),
    }

# -------------------------
# Plotting helpers
# -------------------------
def plot_histogram(aggregates: np.ndarray, bins: int = 200, title: str | None = None, save_path: str | None = None):
    plt.figure(figsize=(9, 5))
    plt.hist(aggregates, bins=bins)
    plt.title(title or "Histogram of Aggregate Losses (linear scale)")
    plt.xlabel("Aggregate loss amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()

def plot_histogram_logx(aggregates: np.ndarray, bins: int = 200, title: str | None = None, save_path: str | None = None):
    plt.figure(figsize=(9, 5))
    plt.hist(aggregates + 1e-12, bins=bins)  # add tiny offset to avoid log(0)
    plt.xscale("log")
    plt.title(title or "Histogram of Aggregate Losses (log x-axis)")
    plt.xlabel("Aggregate loss amount (log scale)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()

def plot_ecdf(aggregates: np.ndarray, title: str | None = None, save_path: str | None = None):
    sorted_vals = np.sort(aggregates)
    ecdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    plt.figure(figsize=(9, 5))
    plt.plot(sorted_vals, ecdf_y)
    plt.title(title or "Empirical CDF of Aggregate Losses")
    plt.xlabel("Aggregate loss amount")
    plt.ylabel("F(x)")
    plt.xlim(left=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()

# -------------------------
# Utilities: saving and summary
# -------------------------
def save_simulation(aggregates: np.ndarray, freqs: np.ndarray, out_dir: str, filename: str = "aggregates.csv") -> str:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({"aggregate_loss": aggregates, "frequency": freqs})
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    return path

def summary_dataframe(metrics: Dict, freqs: np.ndarray, params: Dict) -> pd.DataFrame:
    """Return a neat summary DataFrame of key results and inputs."""
    rows = {
        "Simulations": len(freqs),
        "Mean aggregate loss": metrics["mean"],
        "Std aggregate loss": metrics["std"],
        "Median (VaR 0.5)": float(np.quantile(freqs := None, 0.5)) if False else float(np.nan),  # placeholder removed below
    }

    # We'll compute median of aggregates separately (avoid earlier placeholder)
    # The function will instead accept direct aggregates if needed.
    # We'll build final DataFrame in caller to avoid recompute here.
    return pd.DataFrame.from_dict(rows, orient="index", columns=["Value"])

# -------------------------
# CLI & main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monte Carlo simulation for aggregate insurance claims")
    p.add_argument("--n-sim", type=int, default=100_000, help="Number of Monte Carlo simulations")
    p.add_argument("--lambda", dest="freq_lambda", type=float, default=2.5, help="Poisson frequency lambda")
    p.add_argument("--severity", dest="severity_dist", choices=("lognormal", "gamma"), default="lognormal",
                   help="Severity distribution")
    # Lognormal params
    p.add_argument("--ln-mu", type=float, default=8.5, help="Lognormal mu (mean of underlying normal)")
    p.add_argument("--ln-sigma", type=float, default=1.2, help="Lognormal sigma (sd of underlying normal)")
    # Gamma params
    p.add_argument("--gamma-shape", type=float, default=2.0, help="Gamma shape (k)")
    p.add_argument("--gamma-scale", type=float, default=2000.0, help="Gamma scale (theta)")
    # Misc
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--out-dir", type=str, default="outputs", help="Output directory")
    p.add_argument("--save-plots", action="store_true", help="Save plots to out-dir")
    p.add_argument("--save-csv", action="store_true", help="Save simulated aggregates CSV to out-dir")
    p.add_argument("--bins", type=int, default=200, help="Histogram bins")
    return p.parse_args()

def main():
    args = parse_args()

    params = {
        "n_sim": args.n_sim,
        "freq_lambda": args.freq_lambda,
        "severity_dist": args.severity_dist,
        "lognormal_mu": args.ln_mu,
        "lognormal_sigma": args.ln_sigma,
        "gamma_shape": args.gamma_shape,
        "gamma_scale": args.gamma_scale,
        "random_seed": args.seed,
    }

    print("Running Monte Carlo simulation with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print("Simulating...")

    aggregates, freqs = simulate_aggregate(
        n_sim=params["n_sim"],
        freq_lambda=params["freq_lambda"],
        severity_dist=params["severity_dist"],
        lognormal_mu=params["lognormal_mu"],
        lognormal_sigma=params["lognormal_sigma"],
        gamma_shape=params["gamma_shape"],
        gamma_scale=params["gamma_scale"],
        random_seed=params["random_seed"],
    )

    metrics = risk_metrics(aggregates, alphas=(0.9, 0.95, 0.99))

    # Print summary
    print("\n--- Summary ---")
    print(f"Simulations: {len(aggregates)}")
    print(f"Mean aggregate loss: {metrics['mean']:.6g}")
    print(f"Std aggregate loss: {metrics['std']:.6g}")
    print(f"Median (VaR 50%): {float(np.quantile(aggregates, 0.5)):.6g}")
    for a, v in zip(metrics["alphas"], metrics["VaR"]):
        print(f"VaR {int(a*100)}%: {v:.6g}")
    for a, t in zip(metrics["alphas"], metrics["TVaR"]):
        print(f"TVaR {int(a*100)}%: {t:.6g}")
    print(f"Avg simulated frequency: {freqs.mean():.6g}")
    print(f"Input lambda: {params['freq_lambda']}")

    # Save CSV
    if args.save_csv:
        csv_path = save_simulation(aggregates, freqs, args.out_dir)
        print(f"\nSaved simulation CSV to: {csv_path}")

    # Plotting
    save_prefix = None
    if args.save_plots:
        os.makedirs(args.out_dir, exist_ok=True)
        save_prefix = os.path.join(args.out_dir, "mc_aggregate")

    plot_histogram(aggregates, bins=args.bins, save_path=(save_prefix + "_hist.png" if save_prefix else None))
    plot_ecdf(aggregates, save_path=(save_prefix + "_ecdf.png" if save_prefix else None))
    plot_histogram_logx(aggregates, bins=args.bins, save_path=(save_prefix + "_hist_logx.png" if save_prefix else None))

    # Create a small DataFrame summary and print
    summary = {
        "Simulations": len(aggregates),
        "Mean aggregate loss": metrics["mean"],
        "Std aggregate loss": metrics["std"],
        "Median (VaR 0.5)": float(np.quantile(aggregates, 0.5)),
        "VaR 90%": float(metrics["VaR"][0]),
        "VaR 95%": float(metrics["VaR"][1]),
        "VaR 99%": float(metrics["VaR"][2]),
        "TVaR 90%": float(metrics["TVaR"][0]),
        "TVaR 95%": float(metrics["TVaR"][1]),
        "TVaR 99%": float(metrics["TVaR"][2]),
        "Avg frequency (simulated)": float(freqs.mean()),
        "Lambda (input)": params["freq_lambda"],
        "Severity dist": params["severity_dist"],
    }
    df_summary = pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])
    print("\nSummary table:\n")
    print(df_summary.to_string(float_format='{:,.6g}'.format))

if __name__ == "__main__":
    main()
