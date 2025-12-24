# -*- coding: utf-8 -*-
"""
Convergence plotting utilities for VAE training.
"""
import os
import glob
import pickle
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt


def find_latest_run_dir(results_root: str, fallback: Optional[str] = None) -> str:
    """Find the most recently modified run directory."""
    subdirs = [d for d in glob.glob(os.path.join(results_root, "*")) if os.path.isdir(d)]
    if not subdirs:
        if fallback:
            return fallback
        raise FileNotFoundError(f"No subdirectories under {results_root}")
    return max(subdirs, key=os.path.getmtime)


def extract_fold_histories(summary_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract fold histories from various possible layouts in results_summary.
    
    Supports:
    - summary['vae_training_history']['fold_histories'] (multi-fold)
    - summary['vae_training_history']['history'] (single nested)
    - summary['vae_training_history'] itself (flat history dict)
    - summary['history'] at top level
    """
    vae_hist = summary_dict.get("vae_training_history", None)

    # Case A: {'vae_training_history': {'fold_histories': [...]}}
    if isinstance(vae_hist, dict) and isinstance(vae_hist.get("fold_histories"), list):
        return vae_hist["fold_histories"]

    # Case B: {'vae_training_history': {'history': {...}}}
    if isinstance(vae_hist, dict) and isinstance(vae_hist.get("history"), dict):
        return [vae_hist["history"]]

    # Case C: 'vae_training_history' is a flat history dict
    if isinstance(vae_hist, dict):
        has_train_val = any(k.startswith("train_") or k.startswith("val_") for k in vae_hist.keys())
        if has_train_val:
            return [vae_hist]

    # Case D: Top-level 'history'
    top_hist = summary_dict.get("history", None)
    if isinstance(top_hist, dict):
        return [top_hist]

    raise ValueError(f"Could not locate training histories. Keys: {list(summary_dict.keys())}")


def plot_convergence(fold_histories: List[Dict[str, Any]], 
                     save_path: Optional[str] = None,
                     show: bool = True) -> None:
    """
    Plot train/val loss vs epoch for all folds.
    
    Args:
        fold_histories: List of history dicts, one per fold
        save_path: Optional path to save the figure
        show: Whether to display the figure
    """
    plt.figure(figsize=(8, 5))

    for i, hist in enumerate(fold_histories):
        train_key = next((k for k in ["train_total_loss", "train_loss", "train_elbo"] if k in hist), None)
        val_key = next((k for k in ["val_total_loss", "val_loss", "val_elbo"] if k in hist), None)

        if train_key is None or val_key is None:
            print(f"[Convergence] Fold {i+1}: missing train/val keys; has {list(hist.keys())}")
            continue

        train_losses = np.array(hist[train_key], dtype=float)
        val_losses = np.array(hist[val_key], dtype=float)
        epochs = np.arange(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, alpha=0.3, label="train" if i == 0 else None)
        plt.plot(epochs, val_losses, alpha=0.3, linestyle="--", label="val" if i == 0 else None)

    n_folds = len(fold_histories)
    plt.xlabel("Epoch")
    plt.ylabel("Total loss (≈ -ELBO)")
    plt.title(f"SemiSupMIWAE convergence ({n_folds}-fold CV)" if n_folds > 1 else "SemiSupMIWAE convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def summarize_epochs_per_fold(fold_histories: List[Dict[str, Any]]) -> List[int]:
    """Calculate epochs trained per fold."""
    epochs_per_fold = []
    for hist in fold_histories:
        train_key = next((k for k in ["train_total_loss", "train_loss", "train_elbo"] if k in hist), None)
        if train_key and isinstance(hist[train_key], (list, tuple)):
            epochs_per_fold.append(len(hist[train_key]))
    return epochs_per_fold


def run_convergence_analysis(results_pkl_path: str, save_plot_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point for convergence analysis.
    
    Args:
        results_pkl_path: Path to miwae_analysis_results.pkl
        save_plot_path: Optional path to save convergence plot
    
    Returns:
        Dict with fold_histories and epochs_per_fold
    """
    with open(results_pkl_path, "rb") as f:
        results_summary = pickle.load(f)
    
    fold_histories = extract_fold_histories(results_summary)
    epochs_per_fold = summarize_epochs_per_fold(fold_histories)
    
    if fold_histories:
        plot_convergence(fold_histories, save_path=save_plot_path)
        print(f"[Convergence] Epochs per fold: {epochs_per_fold}")
        print(f"[Convergence] Mean epochs: {np.mean(epochs_per_fold):.1f}")
    
    return {"fold_histories": fold_histories, "epochs_per_fold": epochs_per_fold}


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        run_convergence_analysis(sys.argv[1])
    else:
        print("Usage: python convergence.py <path_to_results.pkl>")
