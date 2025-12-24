# -*- coding: utf-8 -*-
"""
probml.evaluation - Post-training evaluation and visualization utilities.

Entry Points
------------
- `run_convergence_analysis(results_pkl_path)` - Plot training convergence curves
- `run_full_evaluation(run_dir)` - Complete post-training evaluation pipeline

Quick Start::

    from probml.evaluation import run_convergence_analysis, run_full_evaluation
    
    # Convergence plots only
    run_convergence_analysis("results/miwae_analysis_results.pkl")
    
    # Full evaluation with all metrics and plots
    results = run_full_evaluation("results/", output_dir="plots/")

Command Line::

    python -m probml.evaluation.convergence results/miwae_analysis_results.pkl
    python -m probml.evaluation.post_training results/ plots/
"""
from probml.evaluation.convergence import (
    run_convergence_analysis,
    plot_convergence,
    extract_fold_histories,
    find_latest_run_dir,
    summarize_epochs_per_fold,
)
from probml.evaluation.post_training import (
    run_full_evaluation,
    load_results,
    compute_metrics,
    print_metrics,
    plot_residual_histogram,
    plot_residual_qq,
    plot_latent_by_price_decile,
    plot_latent_by_building_class,
    plot_latent_marginals,
    analyze_residual_structure,
)

__all__ = [
    # Main entry points
    'run_convergence_analysis',
    'run_full_evaluation',
    # Convergence utilities
    'plot_convergence',
    'extract_fold_histories',
    'find_latest_run_dir',
    'summarize_epochs_per_fold',
    # Post-training utilities
    'load_results',
    'compute_metrics',
    'print_metrics',
    'plot_residual_histogram',
    'plot_residual_qq',
    'plot_latent_by_price_decile',
    'plot_latent_by_building_class',
    'plot_latent_marginals',
    'analyze_residual_structure',
]
