# -*- coding: utf-8 -*-
from probml.core.utils import get_logger

# =============================================================================
# Latentâ€‘space EDA for a trained VAE (Optimized for Speed & Fixed v2)
# -----------------------------------------------------------------------------
# â€¢ Works with a NumPy array `data_for_encoding` and a Pandas `metadata_df`
# â€¢ Produces KLâ€‘histogram, PCA plots, tâ€‘SNE, kâ€‘means overlay, cosine heatâ€‘map
# â€¢ Saves artefacts into `out_dir`
# â€¢ Incorporates speed optimizations for large datasets.
# â€¢ Fixes OpenTSNE API call.
# â€¢ Fixes TypeError when plotting KMeans results.
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import PCA
# Standard TSNE from sklearn
from sklearn.manifold import TSNE as SKL_TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset
import time

# Attempt to import OpenTSNE for potentially faster t-SNE
try:
    from openTSNE import TSNE as OpenTSNE_TSNE
    from openTSNE import affinity # Usually needed for advanced OpenTSNE usage, good to import
    OPEN_TSNE_AVAILABLE = True
    print("OpenTSNE found, will be used if enabled.")
except ImportError:
    OPEN_TSNE_AVAILABLE = False
    print("OpenTSNE not found, falling back to scikit-learn TSNE.")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Main EDA routine (Optimized & Fixed v2)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_latent_eda(model,
                   data_for_encoding: np.ndarray,
                   metadata_df: pd.DataFrame,
                   year_col_name: str,
                   x_coord_col_name: str,
                   y_coord_col_name: str,
                   device='cuda',
                   out_dir='latent_eda_out',
                   batch_size=1024, # Increased default batch size for faster encoding
                   k_clusters=8,
                   random_seed=42,
                   # Speed/Rigor parameters
                   plot_dpi=150, # Lower DPI for speed, increase for publication
                   tsne_n_samples=5000, # Max samples for t-SNE
                   tsne_n_iter=500,     # Iterations for t-SNE (250-1000 typical)
                   tsne_perplexity=30,
                   tsne_on_pca_components=50, # Use top N PCs for t-SNE (None to use mu_mat)
                   use_opentsne_if_available=True,
                   kmeans_n_init=3, # Reduced from 'auto' or 10 for speed
                   run_tsne_plot=True, # Flag to run t-SNE at all
                   run_cosine_heatmap=True # Flag to run cosine heatmap
                   ):
    """
    Perform exploratory analysis of the VAE latent space (Optimized for Speed & Fixed v2).
    """
    print(f"--- Running Basic Latent EDA (Optimized & Fixed v2) ---")
    print(f"Parameters: tsne_n_samples={tsne_n_samples}, tsne_n_iter={tsne_n_iter}, tsne_on_pca_components={tsne_on_pca_components}, kmeans_n_init={kmeans_n_init}")

    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(random_seed) # For reproducibility in sampling/sklearn
    _device = torch.device(device)
    model = model.to(_device).eval()

    if data_for_encoding.dtype != np.float32:
        data_for_encoding = data_for_encoding.astype(np.float32)

    pytorch_dataset = TensorDataset(torch.from_numpy(data_for_encoding))

    num_loader_workers = 0 # Often safer for speed/compatibility, esp. on Windows & with GPUs
    if _device.type != 'cuda' and hasattr(os, 'sched_getaffinity'): # more workers for CPU processing
         # Check if sched_getaffinity is available and returns a non-empty set
         try:
             affinity_set = os.sched_getaffinity(0)
             if affinity_set:
                 num_loader_workers = len(affinity_set) // 2 if len(affinity_set) > 1 else 0
         except NotImplementedError:
             # os.sched_getaffinity is not implemented on this platform
             num_loader_workers = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 2 else 0


    loader = DataLoader(
        pytorch_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_loader_workers,
        pin_memory=(_device.type == 'cuda')
    )

    print(f"Encoding {data_for_encoding.shape[0]} samples...")
    start_time_encode = time.time()
    mu_list, logvar_list_for_kl = [], [] # Assuming model.encode returns mu, logvar

    # Pre-fetch metadata columns if they exist to speed up the loop
    year_series = metadata_df.get(year_col_name)
    x_coord_series = metadata_df.get(x_coord_col_name)
    y_coord_series = metadata_df.get(y_coord_col_name)

    with torch.no_grad():
        for i, (x_batch_tuple,) in enumerate(loader):
            x_batch = x_batch_tuple.to(_device)
            # Ensure your VAE's encode method returns (mu, logvar)
            # If it only returns mu, then logvar_batch will be problematic.
            mu_batch, logvar_batch = model.encode(x_batch) # ASSUMES (mu, logvar) output
            mu_list.append(mu_batch.cpu())
            logvar_list_for_kl.append(logvar_batch.cpu()) # Collect logvar for KL

    mu_mat = torch.cat(mu_list).numpy()
    logvar_mat = torch.cat(logvar_list_for_kl).numpy()

    # Optimized metadata extraction (after encoding)
    all_indices = np.arange(data_for_encoding.shape[0])
    # Ensure metadata aligns with the number of encoded samples
    if metadata_df.shape[0] != mu_mat.shape[0]:
        print(f"Warning: Metadata rows ({metadata_df.shape[0]}) != Encoded rows ({mu_mat.shape[0]}). Using metadata for first {mu_mat.shape[0]} rows.")
        metadata_df_aligned = metadata_df.iloc[:mu_mat.shape[0]]
    else:
        metadata_df_aligned = metadata_df

    meta_df_out = pd.DataFrame({
        'sample_id': all_indices,
        # Use .values to avoid potential index mismatch issues if series were filtered
        'year_index': metadata_df_aligned[year_col_name].values if year_col_name in metadata_df_aligned else np.nan,
        'grid_x': metadata_df_aligned[x_coord_col_name].values if x_coord_col_name in metadata_df_aligned else np.nan,
        'grid_y': metadata_df_aligned[y_coord_col_name].values if y_coord_col_name in metadata_df_aligned else np.nan,
    }, index=all_indices) # Use original indices if needed later


    print(f"Encoding completed in {time.time() - start_time_encode:.2f}s.")
    print(f'Collected {mu_mat.shape[0]:,} latent vectors (dim {mu_mat.shape[1]}) and logvars.')

    z_cols = [f'z_mu_{d:03d}' for d in range(mu_mat.shape[1])] # Clarify these are mu
    df_lat = pd.concat([meta_df_out.reset_index(drop=True), pd.DataFrame(mu_mat, columns=z_cols)], axis=1)

    # KL Divergence (Standard formula for Gaussian VAE N(0,I) prior)
    kl_values = -0.5 * np.sum(1 + logvar_mat - mu_mat**2 - np.exp(logvar_mat), axis=1)
    df_lat['kl_divergence'] = kl_values

    plt.figure(figsize=(6,4))
    sns.histplot(kl_values, bins=50, kde=True)
    plt.xlabel('Perâ€‘sample KL Divergence'); plt.title('KL Divergence Distribution'); plt.tight_layout()
    plt.savefig(f'{out_dir}/kl_hist.png', dpi=plot_dpi); plt.close()

    print("Running PCA...")
    start_time_pca = time.time()
    n_pca_components = min(min(50, mu_mat.shape[1]), mu_mat.shape[0]) # Use more components if available, cap at 50
    if n_pca_components < 1:
        print(f"Skipping PCA as n_components ({n_pca_components}) is less than 1.")
        pc = np.array([]) # Empty array
        pca_model = None
    else:
        pca_model = PCA(n_components=n_pca_components, random_state=random_seed).fit(mu_mat)
        pc = pca_model.transform(mu_mat)
        plt.figure(figsize=(6,4))
        plt.plot(np.cumsum(pca_model.explained_variance_ratio_)*100, marker='o')
        plt.ylabel('% variance'); plt.xlabel('# PCs'); plt.title('Latent PCA â€“ cumulative')
        plt.grid(); plt.tight_layout(); plt.savefig(f'{out_dir}/pca_cumvar.png', dpi=plot_dpi); plt.close()

        # Check if year_index column exists and has non-NA values before plotting
        if 'year_index' in df_lat.columns and df_lat['year_index'].notna().any() and pc.shape[0] > 0 and pc.shape[1] >= 2:
            plt.figure(figsize=(6,6))
            # Convert year to category for discrete colors, handle potential NaNs
            hue_data = df_lat['year_index'].astype('category') if pd.api.types.is_numeric_dtype(df_lat['year_index']) else df_lat['year_index']
            sns.scatterplot(x=pc[:,0], y=pc[:,1], hue=hue_data,
                            s=10, palette='viridis', legend='auto', alpha=0.7)
            plt.title('PC1 vs PC2');
            plt.legend(title='year', bbox_to_anchor=(1.02,1), loc="upper left")
            plt.tight_layout(); plt.savefig(f'{out_dir}/pca_scatter.png', dpi=plot_dpi); plt.close()
        else:
            print("Skipping PCA scatter plot due to missing/invalid year_index or insufficient PCs/data.")
    print(f"PCA completed in {time.time() - start_time_pca:.2f}s.")


    tsne_embedding = None
    if run_tsne_plot:
        print("Preparing for t-SNE...")
        start_time_tsne = time.time()
        data_for_tsne = mu_mat
        n_samples_original = mu_mat.shape[0]

        indices_tsne = np.arange(n_samples_original)
        if n_samples_original > tsne_n_samples:
            print(f"Subsampling {tsne_n_samples} from {n_samples_original} for t-SNE.")
            indices_tsne = np.random.choice(n_samples_original, tsne_n_samples, replace=False)
            data_for_tsne = mu_mat[indices_tsne, :]

        tsne_input_source = "mu_mat (latent vectors)"
        if tsne_on_pca_components is not None and pc.shape[0] > 0:
             effective_pca_comps = min(tsne_on_pca_components, pc.shape[1])
             if effective_pca_comps > 1:
                  print(f"Running t-SNE on top {effective_pca_comps} PCs.")
                  # Ensure we use the correct indices if data was subsampled
                  data_for_tsne = pc[indices_tsne, :effective_pca_comps]
                  tsne_input_source = f"top {effective_pca_comps} PCs"
             else: print(f"Not enough PCA components ({pc.shape[1]}), using original data for t-SNE.")
        elif tsne_on_pca_components is not None: print("PCA vectors not available, using original data for t-SNE.")

        current_n_samples_for_tsne = data_for_tsne.shape[0]

        # Adjust perplexity: must be less than n_samples
        actual_perplexity = min(tsne_perplexity, current_n_samples_for_tsne - 1)
        if actual_perplexity < 5 and current_n_samples_for_tsne >=5 : actual_perplexity = 5 # Lower bound if possible

        if current_n_samples_for_tsne > 1 and actual_perplexity >= 1 and actual_perplexity < current_n_samples_for_tsne :
            tsne_kwargs = {
                "n_components": 2, "perplexity": actual_perplexity,
                "random_state": random_seed, "n_jobs": -1
            }
            tsne_impl_name = "scikit-learn TSNE"

            if use_opentsne_if_available and OPEN_TSNE_AVAILABLE:
                # OpenTSNE parameters
                tsne_kwargs["n_iter"] = tsne_n_iter
                tsne_kwargs["early_exaggeration_iter"] = max(50, tsne_n_iter // 4)
                tsne_kwargs["n_iter"] = max(100, tsne_n_iter * 3 // 4)
                if "n_jobs" in tsne_kwargs: del tsne_kwargs["n_jobs"]
                tsne_model = OpenTSNE_TSNE(**tsne_kwargs)
                tsne_impl_name = "OpenTSNE"
                print(f"Running {tsne_impl_name} (Perp={actual_perplexity:.1f}, NIter~{tsne_n_iter}) on {current_n_samples_for_tsne} samples ({tsne_input_source})...")
                # --- FIX: Call OpenTSNE's .fit() method ---
                tsne_embedding = tsne_model.fit(data_for_tsne.astype(np.float64))
                # --- End Fix ---
            else: # Fallback to sklearn
                tsne_kwargs["init"] = 'pca'
                tsne_kwargs["learning_rate"] = 'auto'
                tsne_kwargs["n_iter"] = tsne_n_iter
                tsne_model = SKL_TSNE(**tsne_kwargs)
                print(f"Running {tsne_impl_name} (Perp={actual_perplexity:.1f}, NIter={tsne_n_iter}) on {current_n_samples_for_tsne} samples ({tsne_input_source})...")
                tsne_embedding = tsne_model.fit_transform(data_for_tsne.astype(np.float64))

            # Plotting t-SNE
            tsne_meta_df = df_lat.iloc[indices_tsne].reset_index(drop=True) # Use subsampled indices on df_lat
            if 'year_index' in tsne_meta_df.columns and tsne_meta_df['year_index'].notna().any():
                plt.figure(figsize=(7,6))
                hue_data_tsne = tsne_meta_df['year_index'].astype('category') if pd.api.types.is_numeric_dtype(tsne_meta_df['year_index']) else tsne_meta_df['year_index']
                sns.scatterplot(x=tsne_embedding[:,0], y=tsne_embedding[:,1],
                                hue=hue_data_tsne, s=10, palette='viridis', legend='auto', alpha=0.7)
                plt.title(f'{tsne_impl_name} (Perp {actual_perplexity:.0f}, NIter {tsne_n_iter})')
                plt.legend(title='year', bbox_to_anchor=(1.02,1), loc="upper left")
                plt.tight_layout(); plt.savefig(f'{out_dir}/tsne.png', dpi=plot_dpi); plt.close()
            else: # Basic plot if no year data
                plt.figure(figsize=(7,6))
                sns.scatterplot(x=tsne_embedding[:,0], y=tsne_embedding[:,1], s=10, alpha=0.7)
                plt.title(f'{tsne_impl_name} (Perp {actual_perplexity:.0f}, NIter {tsne_n_iter})')
                plt.tight_layout(); plt.savefig(f'{out_dir}/tsne_no_hue.png', dpi=plot_dpi); plt.close()
            print(f"t-SNE completed in {time.time() - start_time_tsne:.2f}s.")
        else:
            print(f'Skipping t-SNE plot due to insufficient samples ({current_n_samples_for_tsne}) or invalid perplexity ({actual_perplexity}).')
    else:
        print("Skipping t-SNE plot as per configuration (run_tsne_plot=False).")

    print("Running K-Means...")
    start_time_kmeans = time.time()
    if pc.shape[0] > 0 and pc.shape[1] >= 2: # K-means overlay on PCA
        actual_k_clusters = min(k_clusters, mu_mat.shape[0])
        if actual_k_clusters > 1:
            kmeans = KMeans(n_clusters=actual_k_clusters, n_init=kmeans_n_init, random_state=random_seed, algorithm="lloyd")
            labels = kmeans.fit_predict(mu_mat) # Cluster on original mu_mat
            df_lat['cluster'] = labels

            plt.figure(figsize=(6,6))
            # --- FIX: Remove .astype('category') ---
            sns.scatterplot(x=pc[:,0], y=pc[:,1], hue=labels, palette='tab10', s=10, legend='auto', alpha=0.7)
            # --- End Fix ---
            plt.title(f'PCA scatter with Kâ€‘means (k={actual_k_clusters})')
            # Add legend if number of clusters is reasonable
            if actual_k_clusters <= 20:
                 plt.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
            else:
                 plt.legend().remove() # Remove legend if too many clusters
            plt.tight_layout(); plt.savefig(f'{out_dir}/pca_kmeans.png', dpi=plot_dpi); plt.close()
        else:
            print(f"Skipping K-means as k_clusters ({actual_k_clusters}) is too small.")
            df_lat['cluster'] = 0
    else:
        print("Skipping K-means overlay on PCA plot due to insufficient PCs/data.")
        df_lat['cluster'] = 0
    print(f"K-Means completed in {time.time() - start_time_kmeans:.2f}s.")

    if run_cosine_heatmap:
        print("Running Cosine Similarity Heatmap...")
        start_time_cosine = time.time()
        n_show_cosine = min(100, mu_mat.shape[0]) # Keep sample size small for clustermap
        if n_show_cosine > 1:
            sub_indices_cosine = np.random.choice(mu_mat.shape[0], n_show_cosine, replace=False)
            sim = cosine_similarity(mu_mat[sub_indices_cosine])
            try:
                g = sns.clustermap(sim, cmap='viridis', figsize=(8,8))
                g.fig.suptitle(f'Cosine similarity â€“ {n_show_cosine} random latent vectors', y=1.02)
                plt.savefig(f'{out_dir}/cosine_heatmap.png', dpi=plot_dpi); plt.close(g.fig)
            except Exception as e_cm:
                print(f"Error during clustermap generation: {e_cm}. Skipping.")
                plt.close() # Ensure figure is closed even if save fails
        else:
            print(f"Skipping cosine similarity heatmap as n_show_cosine ({n_show_cosine}) is too small.")
        print(f"Cosine Similarity completed in {time.time() - start_time_cosine:.2f}s.")
    else:
        print("Skipping Cosine Similarity Heatmap as per configuration.")

    print("Saving artifacts...")
    df_lat.to_parquet(f'{out_dir}/latent_table.parquet')
    if pc.shape[0] > 0: np.save(f'{out_dir}/pc_top_components.npy', pc)
    if tsne_embedding is not None: np.save(f'{out_dir}/tsne_embedding_sampled.npy', tsne_embedding) # Note: this is on sampled data

    print(f'All basic EDA artefacts saved to â€œ{out_dir}/â€')
    print(f"--- Basic Latent EDA (Optimized & Fixed v2) Finished ---")
    return df_lat

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# EXAMPLE USAGE (to be adapted and placed in the new pipeline script)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# This part needs to be run within the new pipeline script where
# vae_model, vae_input_data, and df_for_vae_prep (or equivalent metadata DF)
# and coordinate/year column names are defined.

# Example placeholder for how it would be called:
# Make sure these variables are defined from your new pipeline:
# pipeline_vae_model: The trained VAE model from your new pipeline.
# pipeline_vae_input_data: The NumPy array used as input for the VAE (e.g., scaled features).
# pipeline_metadata_df: The Pandas DataFrame aligned with `pipeline_vae_input_data`,
#                       containing original features and metadata.
# year_column_for_eda_actual: String name of the year column in `pipeline_metadata_df`.
# actual_x_coord: String name of the X coordinate column in `pipeline_metadata_df`.
# actual_y_coord: String name of the Y coordinate column in `pipeline_metadata_df`.
# DEVICE: 'cuda' or 'cpu'
# VAE_BATCH_SIZE: Batch size used in pipeline
# RANDOM_SEED: Seed used in pipeline
# EDA_PARAMS: Dictionary containing speed parameters like {'plot_dpi': 150, 'tsne_n_samples': 5000, ...}

# if 'pipeline_vae_model' in locals() and 'pipeline_vae_input_data' in locals() and 'pipeline_metadata_df' in locals():
#     df_latent_output = run_latent_eda(
#         model=pipeline_vae_model,
#         data_for_encoding=pipeline_vae_input_data,
#         metadata_df=pipeline_metadata_df,
#         year_col_name=year_column_for_eda_actual,
#         x_coord_col_name=actual_x_coord,
#         y_coord_col_name=actual_y_coord,
#         device=DEVICE,
#         out_dir='latent_eda_output_v1_optimized_fixed_v2', # Updated dir name
#         batch_size=VAE_BATCH_SIZE,
#         random_seed=RANDOM_SEED,
#         **EDA_PARAMS # Pass the dictionary of speed parameters
#     )
#     print(df_latent_output.head())
# else:
#     print("Required variables for run_latent_eda not found.")

# =============================================================================
# Enhanced Latentâ€‘space EDA for a trained VAE (Optimized for Speed & Fixed v2)
# -----------------------------------------------------------------------------
# â€¢ Works with a NumPy array `data_for_encoding` and a Pandas `metadata_df`
# â€¢ Produces comprehensive latent space analysis with advanced visualizations
# â€¢ Includes statistical analysis, clustering insights, advanced plots
# â€¢ Incorporates speed optimizations (Cluster Eval, t-SNE, Latent Traversal).
# â€¢ Fixes NumPy 2.0 compatibility issue (`np.float_`).
# =============================================================================
import os
import time # Import time for benchmarking sections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import PCA
# Standard TSNE from sklearn
from sklearn.manifold import TSNE as SKL_TSNE
# Import KMeans and potentially MiniBatchKMeans
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import kurtosis, skew
from scipy.cluster.hierarchy import dendrogram, linkage
from torch.utils.data import DataLoader, TensorDataset
# from umap import UMAP # Kept commented
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pickle
import json
from datetime import datetime
import warnings # To suppress specific warnings if needed

# Attempt to import OpenTSNE for potentially faster t-SNE
try:
    from openTSNE import TSNE as OpenTSNE_TSNE
    from openTSNE import affinity # Usually needed for advanced OpenTSNE usage, good to import
    OPEN_TSNE_AVAILABLE = True
    print("OpenTSNE found, will be used if enabled.")
except ImportError:
    OPEN_TSNE_AVAILABLE = False
    print("OpenTSNE not found, falling back to scikit-learn TSNE.")

# Set styling for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2) # Slightly smaller font scale might render faster
PALETTE = "viridis"
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_viridis",
                                                plt.cm.viridis(np.linspace(0.1, 0.9, 256)))

# --- Helper to suppress specific warnings during KMeans if needed ---
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings("ignore", category=ConvergenceWarning)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Extended statistical analysis functions (Unchanged from previous fix)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def analyze_latent_statistics(mu_mat):
    """Perform detailed statistical analysis of latent dimensions"""
    n_dims = mu_mat.shape[1]
    with warnings.catch_warnings():
         warnings.simplefilter("ignore", category=RuntimeWarning)
         stats = {
             'mean': np.mean(mu_mat, axis=0), 'std': np.std(mu_mat, axis=0),
             'min': np.min(mu_mat, axis=0), 'max': np.max(mu_mat, axis=0),
             'median': np.median(mu_mat, axis=0), 'skewness': skew(mu_mat, axis=0),
             'kurtosis': kurtosis(mu_mat, axis=0), 'iqr': np.percentile(mu_mat, 75, axis=0) - np.percentile(mu_mat, 25, axis=0),
             'variance': np.var(mu_mat, axis=0), 'activation_ratio': np.mean(np.abs(mu_mat) > 0.1, axis=0)
         }
    stats_df = pd.DataFrame({k: stats[k] for k in stats}, index=[f'z{i:03d}' for i in range(n_dims)])
    stats_df['importance_score'] = ( stats_df['std'].fillna(0) + np.abs(stats_df['skewness'].fillna(0)) * 0.2 + stats_df['activation_ratio'].fillna(0) * 5 )
    corr_mat = np.corrcoef(mu_mat.T)
    return stats_df, corr_mat

def evaluate_clustering_quality(
        mu_mat: np.ndarray, labels: np.ndarray, k_range: range = range(2, 15),
        kmeans_algo_eval: str = 'KMeans', kmeans_n_init_eval: int = 1,
        metric_subsample_size: int = 2000, random_state: int = 42
    ) -> pd.DataFrame:
    """ Evaluate clustering quality metrics across multiple k values (Optimized). """
    print(f"--- Evaluating Cluster Quality (k={list(k_range)}, algo='{kmeans_algo_eval}', n_init={kmeans_n_init_eval}, metric_subsample={metric_subsample_size}) ---")
    start_time = time.time()
    metrics = {'k': [], 'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': []}
    n_samples_total = mu_mat.shape[0]
    if n_samples_total <= 1: return pd.DataFrame(metrics)
    unique_labels_provided, counts = np.unique(labels, return_counts=True)
    n_labels_provided = len(unique_labels_provided)
    actual_metric_subsample_size = min(metric_subsample_size, n_samples_total) if n_samples_total > 1 else None

    def safe_metric_call(metric_fn, data, current_labels, **kwargs):
        try:
            num_unique_labs = len(np.unique(current_labels))
            if 1 < num_unique_labs < n_samples_total:
                call_kwargs = kwargs.copy()
                if metric_fn == silhouette_score:
                    if actual_metric_subsample_size and actual_metric_subsample_size < n_samples_total:
                        call_kwargs['sample_size'] = actual_metric_subsample_size
                        call_kwargs['random_state'] = random_state
                return metric_fn(data, current_labels, **call_kwargs)
            return np.nan
        except ValueError: return np.nan

    if n_labels_provided > 1:
        metrics['k'].append(n_labels_provided)
        metrics['silhouette'].append(safe_metric_call(silhouette_score, mu_mat, labels))
        metrics['calinski_harabasz'].append(safe_metric_call(calinski_harabasz_score, mu_mat, labels))
        metrics['davies_bouldin'].append(safe_metric_call(davies_bouldin_score, mu_mat, labels))

    ClusterImpl = KMeans if kmeans_algo_eval == 'KMeans' else MiniBatchKMeans
    for k_val in k_range:
        if k_val == n_labels_provided: continue
        if k_val >= n_samples_total or k_val <= 1: continue
        try:
             if kmeans_algo_eval == 'KMeans': cluster_model = ClusterImpl(n_clusters=k_val, n_init=kmeans_n_init_eval, random_state=random_state, algorithm="lloyd")
             else: mbk_batch_size = min(1024*3, n_samples_total // 2 if n_samples_total > 1 else 1); cluster_model = ClusterImpl(n_clusters=k_val, n_init=kmeans_n_init_eval, random_state=random_state, batch_size=mbk_batch_size, max_iter=100)
             current_iter_labels = cluster_model.fit_predict(mu_mat)
             metrics['k'].append(k_val)
             metrics['silhouette'].append(safe_metric_call(silhouette_score, mu_mat, current_iter_labels))
             metrics['calinski_harabasz'].append(safe_metric_call(calinski_harabasz_score, mu_mat, current_iter_labels))
             metrics['davies_bouldin'].append(safe_metric_call(davies_bouldin_score, mu_mat, current_iter_labels))
        except Exception as e: print(f"Warning: Clustering/Metric failed for k={k_val}. Error: {e}"); metrics['k'].append(k_val); metrics['silhouette'].append(np.nan); metrics['calinski_harabasz'].append(np.nan); metrics['davies_bouldin'].append(np.nan)
    print(f"Cluster quality evaluation completed in {time.time() - start_time:.2f}s.")
    return pd.DataFrame(metrics).sort_values(by='k').reset_index(drop=True)

def generate_cluster_profiles(mu_mat, logvar_mat, labels, meta_df, year_col_name, x_coord_col_name, y_coord_col_name):
    """Generate statistical profiles for each cluster (Generally fast)"""
    unique_labels = np.unique(labels); profiles = []
    kl_per_sample = -0.5 * np.sum(1 + logvar_mat - mu_mat**2 - np.exp(logvar_mat), axis=1)
    temp_df_data = {'label': labels, 'kl': kl_per_sample}
    if year_col_name and year_col_name in meta_df.columns: temp_df_data[year_col_name] = meta_df[year_col_name]
    if x_coord_col_name and x_coord_col_name in meta_df.columns: temp_df_data[x_coord_col_name] = meta_df[x_coord_col_name]
    if y_coord_col_name and y_coord_col_name in meta_df.columns: temp_df_data[y_coord_col_name] = meta_df[y_coord_col_name]
    temp_df = pd.DataFrame(temp_df_data)
    mu_df = pd.DataFrame(mu_mat, columns=[f'mu_{i}' for i in range(mu_mat.shape[1])])
    temp_df = pd.concat([temp_df.reset_index(drop=True), mu_df.reset_index(drop=True)], axis=1)
    grouped = temp_df.groupby('label')
    for label, group in grouped:
        cluster_mu_data = group[[f'mu_{i}' for i in range(mu_mat.shape[1])]].values
        profile = {'cluster_id': label, 'size': len(group), 'proportion': len(group) / len(labels), 'center_mu': np.mean(cluster_mu_data, axis=0), 'dispersion_mu': np.mean(np.std(cluster_mu_data, axis=0)), 'avg_kl': group['kl'].mean()}
        if year_col_name and year_col_name in group.columns: profile['year_distribution'] = group[year_col_name].value_counts(normalize=True).to_dict()
        spatial_dist = {}
        if x_coord_col_name and x_coord_col_name in group.columns: spatial_dist['x_mean'] = group[x_coord_col_name].mean(); spatial_dist['x_std'] = group[x_coord_col_name].std()
        if y_coord_col_name and y_coord_col_name in group.columns: spatial_dist['y_mean'] = group[y_coord_col_name].mean(); spatial_dist['y_std'] = group[y_coord_col_name].std()
        if spatial_dist: profile['spatial_distribution'] = spatial_dist
        profiles.append(profile)
    return profiles

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Advanced visualization functions (Optimized & Fixed)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def plot_kl_analysis(kl_per_sample, out_dir, plot_dpi=150):
    """Generate comprehensive KL divergence visualizations"""
    log_kl = np.log1p(np.abs(kl_per_sample)); fig = plt.figure(figsize=(18, 6)); gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    ax0 = plt.subplot(gs[0]); sns.histplot(kl_per_sample, bins=50, kde=True, color='darkblue', ax=ax0); ax0.set_xlabel('KL Divergence per Sample'); ax0.set_ylabel('Frequency'); ax0.set_title('KL Distribution')
    ax1 = plt.subplot(gs[1]); sns.histplot(log_kl, bins=50, kde=True, color='darkgreen', ax=ax1); ax1.set_xlabel('Log(1+|KL|)'); ax1.set_ylabel('Frequency'); ax1.set_title('Log-Transformed KL Distribution')
    ax2 = plt.subplot(gs[2]); sns.ecdfplot(kl_per_sample, ax=ax2, color='darkred'); ax2.set_xlabel('KL Divergence per Sample'); ax2.set_ylabel('Cumulative Proportion'); ax2.set_title('Cumulative KL Distribution')
    plt.tight_layout(); plt.savefig(f'{out_dir}/kl_analysis.png', dpi=plot_dpi); plt.close()
    return kl_per_sample

def plot_pca_analysis(mu_mat, meta_df, labels, year_col_name, out_dir, plot_dpi=150):
    """Generate comprehensive PCA visualizations (NumPy 2.0 Fix)"""
    start_time = time.time(); pca = PCA(n_components=min(mu_mat.shape[1], mu_mat.shape[0], 100), random_state=42).fit(mu_mat)
    pc = pca.transform(mu_mat); explained_variance = pca.explained_variance_ratio_
    print(f"PCA calculation took {time.time() - start_time:.2f}s")
    n_pcs_to_show = min(10, pca.n_components_); pc_df = pd.DataFrame(pc[:, :n_pcs_to_show], columns=[f'PC{i+1}' for i in range(n_pcs_to_show)])
    pc_df = pd.concat([meta_df.reset_index(drop=True), pc_df], axis=1); pc_df['cluster'] = labels
    fig, ax = plt.subplots(figsize=(12, 6)); num_components_plot = pca.n_components_
    ax.bar(range(1, num_components_plot + 1), explained_variance * 100, color=plt.cm.viridis(np.linspace(0, 1, num_components_plot))); ax.plot(range(1, num_components_plot + 1), np.cumsum(explained_variance) * 100, 'o-', color='darkred', linewidth=3)
    ax.set_xlabel('Principal Component'); ax.set_ylabel('Explained Variance (%)'); ax.set_title('PCA Explained Variance'); ax.grid(True, linestyle='--', alpha=0.7)
    ax2 = ax.twinx(); ax2.set_ylabel('Cumulative Explained Variance (%)', color='darkred'); ax2.tick_params(axis='y', labelcolor='darkred')
    max_ev = np.nanmax(explained_variance); ax.set_ylim(0, max_ev * 100 * 1.1 if np.isfinite(max_ev) and max_ev > 0 else 10); ax2.set_ylim(0, 105); ax.set_xlim(0.5, num_components_plot + 0.5)
    plt.tight_layout(); plt.savefig(f'{out_dir}/pca_variance.png', dpi=plot_dpi); plt.close()
    hover_data_cols = ['sample_id']; # Start with sample_id
    if 'grid_x' in pc_df.columns: hover_data_cols.append('grid_x')
    if 'grid_y' in pc_df.columns: hover_data_cols.append('grid_y')
    if year_col_name and year_col_name in pc_df.columns: hover_data_cols.append(year_col_name)
    is_cluster_float = np.issubdtype(pc_df['cluster'].dtype, np.floating) if 'cluster' in pc_df.columns else False
    if 'PC1' in pc_df.columns and 'PC2' in pc_df.columns:
        fig_pca_2d = px.scatter(pc_df, x='PC1', y='PC2', color='cluster' if 'cluster' in pc_df.columns else None, symbol=year_col_name if year_col_name and year_col_name in pc_df.columns else None, hover_data=hover_data_cols, color_continuous_scale='viridis' if is_cluster_float else None, color_discrete_map={-1: "lightgrey"} if -1 in labels else None, title='PCA Visualization (PC1 vs PC2)'); fig_pca_2d.write_html(f'{out_dir}/pca_interactive.html')
    if 'PC3' in pc_df.columns:
        fig_pca_3d = px.scatter_3d(pc_df, x='PC1', y='PC2', z='PC3', color='cluster' if 'cluster' in pc_df.columns else None, symbol=year_col_name if year_col_name and year_col_name in pc_df.columns else None, hover_data=hover_data_cols, color_continuous_scale='viridis' if is_cluster_float else None, color_discrete_map={-1: "lightgrey"} if -1 in labels else None, title='3D PCA Visualization'); fig_pca_3d.write_html(f'{out_dir}/pca_3d.html')
    num_pcs_pairplot = min(4, n_pcs_to_show)
    if num_pcs_pairplot > 1 and year_col_name and year_col_name in pc_df.columns: sns.pairplot(pc_df, vars=[f'PC{i+1}' for i in range(num_pcs_pairplot)], hue=year_col_name, palette='viridis', plot_kws={'alpha': 0.5, 's': 10}, diag_kind='kde'); plt.savefig(f'{out_dir}/pca_pairplot.png', dpi=plot_dpi); plt.close()
    if hasattr(pca, 'components_'):
        loadings = pca.components_.T[:, :n_pcs_to_show]; plt.figure(figsize=(max(8, n_pcs_to_show * 0.8), max(6, mu_mat.shape[1] * 0.3))); sns.heatmap(loadings, cmap='coolwarm', center=0, annot=False, fmt=".2f", yticklabels=[f'z{i:03d}' for i in range(loadings.shape[0])], xticklabels=[f'PC{i+1}' for i in range(loadings.shape[1])]); plt.title(f'PCA Loadings (Top {n_pcs_to_show} PCs)'); plt.tight_layout(); plt.savefig(f'{out_dir}/pca_loadings.png', dpi=plot_dpi); plt.close()
    return pc, pca

# def plot_umap_visualization(...): # Kept commented

def plot_clustering_analysis(mu_mat, labels, out_dir, cluster_metrics, cluster_profiles, plot_dpi=150, dendrogram_n_samples=2000):
    """Generate comprehensive clustering analysis visualizations (Optimized Dendrogram)"""
    if not cluster_metrics.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5)); plot_metrics = [('silhouette', 'Silhouette Score', 'Higher is better', 'blue'), ('calinski_harabasz', 'Calinski-Harabasz Score', 'Higher is better', 'green'), ('davies_bouldin', 'Davies-Bouldin Score', 'Lower is better', 'red')]
        for i, (metric_name, title_name, subtitle, color) in enumerate(plot_metrics):
             if metric_name in cluster_metrics.columns: axes[i].plot(cluster_metrics['k'], cluster_metrics[metric_name], 'o-', linewidth=2, color=color); axes[i].set_xlabel('Number of Clusters (k)'); axes[i].set_ylabel(title_name); axes[i].set_title(f'{title_name} by k\n({subtitle})'); axes[i].grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(); plt.savefig(f'{out_dir}/cluster_evaluation.png', dpi=plot_dpi); plt.close()
    else: print("Cluster metrics DataFrame is empty. Skipping cluster evaluation plot.")
    if cluster_profiles:
         cluster_sizes = np.array([p['size'] for p in cluster_profiles if 'size' in p]); cluster_kl = np.array([p['avg_kl'] for p in cluster_profiles if 'avg_kl' in p]); cluster_dispersion = np.array([p['dispersion_mu'] for p in cluster_profiles if 'dispersion_mu' in p])
         if len(cluster_sizes) > 0 and len(cluster_kl) > 0 and len(cluster_dispersion) > 0:
              fig, ax = plt.subplots(figsize=(12, 6)); size_norm = cluster_sizes / max(1, max(cluster_sizes))
              scatter = ax.scatter(range(len(cluster_profiles)), cluster_kl, s=size_norm * 500 + 50, c=cluster_dispersion, cmap='viridis', alpha=0.7)
              for i, profile in enumerate(cluster_profiles): ax.text(i, profile['avg_kl'], str(profile['cluster_id']), ha='center', va='center', fontsize=10, fontweight='bold')
              ax.set_xlabel('Cluster ID (index)'); ax.set_xticks(range(len(cluster_profiles))); ax.set_xticklabels([p['cluster_id'] for p in cluster_profiles]); ax.set_ylabel('Average KL Divergence'); ax.set_title('Cluster Profiles: Size, KL, and Dispersion (mu-space)')
              cbar = plt.colorbar(scatter); cbar.set_label('Dispersion (Avg Std Dev in mu-space)')
              plt.tight_layout(); plt.savefig(f'{out_dir}/cluster_profiles.png', dpi=plot_dpi); plt.close()
         else: print("Insufficient data in cluster_profiles to plot profiles visualization.")
    if dendrogram_n_samples > 0:
        print(f"Running Dendrogram linkage calculation on sample size: {dendrogram_n_samples}"); start_time_dendro = time.time()
        actual_dendro_samples = min(dendrogram_n_samples, mu_mat.shape[0])
        if actual_dendro_samples > 1:
            indices_dendro = np.random.choice(mu_mat.shape[0], actual_dendro_samples, replace=False); sample_data_dendro = mu_mat[indices_dendro]
            if sample_data_dendro.shape[0] > 1:
                try:
                    Z = linkage(sample_data_dendro, method='ward'); print(f"Linkage calculation took {time.time() - start_time_dendro:.2f}s")
                    plt.figure(figsize=(12, 8)); p_truncate = min(30, sample_data_dendro.shape[0]-1 if sample_data_dendro.shape[0]>1 else 1)
                    dendrogram(Z, truncate_mode='lastp', p=p_truncate, leaf_rotation=90.); plt.title(f'Hierarchical Clustering Dendrogram (Ward, Sample={actual_dendro_samples})'); plt.xlabel(f'Samples (showing last {p_truncate} merged clusters)'); plt.ylabel('Distance'); plt.tight_layout(); plt.savefig(f'{out_dir}/dendrogram.png', dpi=plot_dpi); plt.close()
                except Exception as e_linkage: print(f"Error during linkage/dendrogram: {e_linkage}")
            else: print("Not enough samples for dendrogram after random choice.")
        else: print("Not enough samples for dendrogram.")
    else: print("Skipping dendrogram plot as per configuration.")

def plot_advanced_similarity_analysis(mu_mat, labels, stats_df, corr_mat, out_dir, plot_dpi=150):
    """Generate advanced similarity and correlation visualizations"""
    if not stats_df.empty and 'importance_score' in stats_df.columns:
         top_dims_series = stats_df.sort_values('importance_score', ascending=False).index[:20]; dim_indices = [int(dim[1:]) for dim in top_dims_series if dim[1:].isdigit()]
         if dim_indices:
              top_corr = corr_mat[np.ix_(dim_indices, dim_indices)]; plt.figure(figsize=(max(10, len(dim_indices)*0.6), max(8, len(dim_indices)*0.5))); sns.heatmap(top_corr, cmap='coolwarm', center=0, annot=False, fmt='.2f', xticklabels=[stats_df.index[i] for i in dim_indices], yticklabels=[stats_df.index[i] for i in dim_indices]); plt.title(f'Correlation Matrix for Top {len(dim_indices)} Important Dimensions'); plt.tight_layout(); plt.savefig(f'{out_dir}/top_dims_correlation.png', dpi=plot_dpi); plt.close()
              top_dims_df = stats_df.loc[[stats_df.index[i] for i in dim_indices]]; plt.figure(figsize=(15, 10)); gs_stats = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.5]); ax0 = plt.subplot(gs_stats[0]); ax0.errorbar(range(len(top_dims_df)), top_dims_df['mean'], yerr=top_dims_df['std'], fmt='o', capsize=5, color='darkblue', ecolor='lightblue'); ax0.set_ylabel('Mean Value (Â± std)'); ax0.set_title('Mean Values of Top Dimensions'); ax0.set_xticks(range(len(top_dims_df))); ax0.set_xticklabels(top_dims_df.index, rotation=45, ha="right"); ax0.grid(True, linestyle='--', alpha=0.7); ax1 = plt.subplot(gs_stats[1]); bar_width = 0.35; positions = np.arange(len(top_dims_df)); ax1.bar(positions - bar_width/2, top_dims_df['skewness'], bar_width, label='Skewness', color='orange'); ax1.bar(positions + bar_width/2, top_dims_df['kurtosis'], bar_width, label='Kurtosis', color='green'); ax1.set_ylabel('Value'); ax1.set_title('Skewness and Kurtosis'); ax1.set_xticks(positions); ax1.set_xticklabels(top_dims_df.index, rotation=45, ha="right"); ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.7); ax2 = plt.subplot(gs_stats[2]); pos_act = np.arange(len(top_dims_df)); ax2.bar(pos_act, top_dims_df['importance_score'], color='purple', alpha=0.7, label='Importance Score'); ax2_twin = ax2.twinx(); ax2_twin.plot(pos_act, top_dims_df['activation_ratio'], 'ro-', linewidth=2, label='Activation Ratio'); ax2.set_xlabel('Dimension'); ax2.set_ylabel('Importance Score', color='purple'); ax2_twin.set_ylabel('Activation Ratio', color='red'); ax2.set_title('Importance and Activation'); ax2.set_xticks(pos_act); ax2.set_xticklabels(top_dims_df.index, rotation=45, ha="right"); lines1, labels1 = ax2.get_legend_handles_labels(); lines2, labels2 = ax2_twin.get_legend_handles_labels(); ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right'); ax2.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(h_pad=3.0); plt.savefig(f'{out_dir}/dimension_statistics.png', dpi=plot_dpi); plt.close()
         else: print("No valid top dimensions found for correlation/stats plots.")
    else: print("Stats DataFrame is empty or missing 'importance_score'. Skipping some advanced similarity plots.")
    n_show_cosine = min(100, mu_mat.shape[0])
    if n_show_cosine > 1:
        sub_indices_cosine = np.random.choice(mu_mat.shape[0], n_show_cosine, replace=False); sim = cosine_similarity(mu_mat[sub_indices_cosine])
        try: g = sns.clustermap(sim, cmap=CUSTOM_CMAP, standard_scale=None, figsize=(10,10), cbar_kws={"label": "Cosine Similarity"}, xticklabels=False, yticklabels=False); g.fig.suptitle(f'Cosine Similarity Heatmap - {n_show_cosine} Random Samples', y=1.02, fontsize=14); plt.savefig(f'{out_dir}/enhanced_cosine_heatmap.png', dpi=plot_dpi); plt.close(g.fig)
        except Exception as e: print(f"Clustermap failed: {e}"); plt.close() # Ensure figure is closed
    else: print(f"Skipping cosine similarity heatmap as n_show_cosine ({n_show_cosine}) is too small.")
    return mu_mat

def plot_spatial_year_analysis(mu_mat, logvar_mat, meta_df, labels, pc, year_col_name, x_coord_col_name, y_coord_col_name, out_dir, plot_dpi=150):
    """Analyze relationship between latent space and spatial/temporal features"""
    kl_per_sample = -0.5 * np.sum(1 + logvar_mat - mu_mat**2 - np.exp(logvar_mat), axis=1)
    analysis_df_data = {'cluster': labels, 'kl_divergence': kl_per_sample}
    for col_name, data_series in [(year_col_name, meta_df.get(year_col_name)),(x_coord_col_name, meta_df.get(x_coord_col_name)),(y_coord_col_name, meta_df.get(y_coord_col_name))]:
         if data_series is not None: analysis_df_data[col_name] = data_series.values
    if pc is not None and pc.shape[0] == len(labels):
         for i in range(min(pc.shape[1], 3)): analysis_df_data[f'PC{i+1}'] = pc[:, i]
    analysis_df = pd.DataFrame(analysis_df_data); analysis_df = analysis_df.reset_index(drop=True)
    if x_coord_col_name in analysis_df.columns and y_coord_col_name in analysis_df.columns:
        plt.figure(figsize=(10, 8)); scatter_spatial = plt.scatter(analysis_df[x_coord_col_name], analysis_df[y_coord_col_name], c=analysis_df['cluster'], cmap='tab10', alpha=0.6, s=8); plt.colorbar(scatter_spatial, label='Cluster'); plt.title('Spatial Distribution of Clusters'); plt.xlabel(x_coord_col_name); plt.ylabel(y_coord_col_name); plt.tight_layout(); plt.savefig(f'{out_dir}/spatial_clusters.png', dpi=plot_dpi); plt.close()
        fig_spatial_plotly = px.scatter(analysis_df, x=x_coord_col_name, y=y_coord_col_name, color='cluster', symbol=year_col_name if year_col_name in analysis_df.columns else None, size='kl_divergence' if 'kl_divergence' in analysis_df.columns else None, hover_data=[col for col in ['PC1', 'PC2', 'kl_divergence'] if col in analysis_df.columns], title='Spatial Distribution (Interactive)', color_discrete_map={-1: "lightgrey"}); fig_spatial_plotly.write_html(f'{out_dir}/spatial_interactive.html')
    if year_col_name in analysis_df.columns:
        year_cluster_counts = analysis_df.groupby([year_col_name, 'cluster']).size().unstack(fill_value=0)
        if not year_cluster_counts.empty: year_cluster_prop = year_cluster_counts.div(year_cluster_counts.sum(axis=1), axis=0); year_cluster_prop.plot(kind='bar', stacked=True, colormap='tab10', figsize=(12,7)); plt.title('Cluster Distribution by Year'); plt.xlabel(year_col_name); plt.ylabel('Proportion'); plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(); plt.savefig(f'{out_dir}/temporal_clusters.png', dpi=plot_dpi); plt.close()
        sns.catplot(x=year_col_name, y='kl_divergence', data=analysis_df, kind="box", palette='viridis', height=5, aspect=2); plt.title('KL Divergence by Year'); plt.xlabel(year_col_name); plt.ylabel('KL Divergence'); plt.xticks(rotation=45, ha='right'); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.savefig(f'{out_dir}/kl_by_year.png', dpi=plot_dpi); plt.close()
    return analysis_df

# --- OPTIMIZED Latent Traversal Plotting ---
def plot_latent_traversal(
        mu_mat: np.ndarray,
        stats_df: pd.DataFrame,
        out_dir: str,
        plot_dpi: int = 150,
        traversal_sample_size: int = 5000, # Subsample for plotting
        n_dims_traversal: int = 5, # Plot top N dims
        n_pairs_traversal: int = 3  # Plot top N pairs
    ):
    """
    Generate latent traversal visualization using histograms (Optimized for Speed).
    Plots distributions (histograms) of top important dimensions and their joint distributions.
    Uses subsampling for faster plotting.
    """
    print(f"--- Plotting Latent Traversal (Sample Size: {traversal_sample_size}, Dims: {n_dims_traversal}, Pairs: {n_pairs_traversal}) ---")
    start_time = time.time()
    if stats_df.empty or 'importance_score' not in stats_df.columns:
        print("Stats DataFrame is empty or missing 'importance_score'. Skipping latent traversal plots.")
        return
    if mu_mat.shape[0] == 0:
        print("Mu matrix is empty. Skipping latent traversal plots.")
        return

    # Subsample data for plotting
    n_total_samples = mu_mat.shape[0]
    actual_sample_size = min(traversal_sample_size, n_total_samples)
    if actual_sample_size < n_total_samples:
        print(f"Subsampling {actual_sample_size} points for traversal plots.")
        sample_indices = np.random.choice(n_total_samples, actual_sample_size, replace=False)
        mu_mat_sampled = mu_mat[sample_indices, :]
    else:
        mu_mat_sampled = mu_mat

    # Get top dimensions based on importance score
    top_dims_series = stats_df.sort_values('importance_score', ascending=False).index[:n_dims_traversal]
    dim_indices = [int(dim[1:]) for dim in top_dims_series if dim.startswith('z') and dim[1:].isdigit()]

    if not dim_indices:
        print("No valid top dimensions found for latent traversal plots.")
        return

    # Plot 1D Histograms for top dimensions
    n_dims_to_plot = len(dim_indices)
    fig_1d, axes_1d = plt.subplots(n_dims_to_plot, 1, figsize=(10, n_dims_to_plot * 2.5), squeeze=False) # Ensure axes_1d is always 2D
    axes_1d = axes_1d.flatten() # Flatten to 1D array for easy iteration

    for i, original_dim_idx in enumerate(dim_indices):
        dim_name = stats_df.index[original_dim_idx]
        # Use histplot instead of kdeplot for speed
        sns.histplot(mu_mat_sampled[:, original_dim_idx], ax=axes_1d[i], bins=50, kde=False, color=plt.cm.viridis(i/n_dims_to_plot))
        axes_1d[i].set_title(f'Dimension {dim_name} (z{original_dim_idx:03d}) Distribution (Sampled)')
        axes_1d[i].set_xlabel('Value')
        axes_1d[i].set_ylabel('Frequency')
        axes_1d[i].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/top_dimensions_distribution_hist.png', dpi=plot_dpi); plt.close(fig_1d)

    # Plot 2D Histograms for top pairs
    n_top_for_pairs = min(n_pairs_traversal, len(dim_indices))
    if n_top_for_pairs >= 2:
        top_pairs_indices = [(dim_indices[i], dim_indices[j]) for i in range(n_top_for_pairs)
                             for j in range(i+1, n_top_for_pairs)]

        n_pairs_to_plot = len(top_pairs_indices)
        if n_pairs_to_plot > 0:
            n_cols_pairs = min(2, n_pairs_to_plot) # Max 2 columns
            n_rows_pairs = (n_pairs_to_plot + n_cols_pairs - 1) // n_cols_pairs

            fig_joint, axes_joint = plt.subplots(n_rows_pairs, n_cols_pairs, figsize=(12, n_rows_pairs * 5), squeeze=False)
            axes_joint = axes_joint.flatten()

            for i, (dim1_idx, dim2_idx) in enumerate(top_pairs_indices):
                if i < len(axes_joint):
                    dim1_name = stats_df.index[dim1_idx]
                    dim2_name = stats_df.index[dim2_idx]
                    # Use histplot (2D) or hist2d for speed
                    sns.histplot(
                        x=mu_mat_sampled[:, dim1_idx],
                        y=mu_mat_sampled[:, dim2_idx],
                        ax=axes_joint[i],
                        bins=30, # Fewer bins for 2D
                        cmap="viridis"
                        # Consider adding cbar=True if needed
                    )
                    axes_joint[i].set_title(f'{dim1_name} vs {dim2_name} (Sampled)')
                    axes_joint[i].set_xlabel(dim1_name)
                    axes_joint[i].set_ylabel(dim2_name)

            for j in range(n_pairs_to_plot, len(axes_joint)): # Hide unused subplots
                if j < len(axes_joint): axes_joint[j].axis('off')

            plt.tight_layout()
            plt.savefig(f'{out_dir}/top_dimensions_joint_hist.png', dpi=plot_dpi); plt.close(fig_joint)
        else: print("Not enough pairs of top dimensions to plot joint distributions.")
    else: print("Not enough top dimensions (need at least 2) for joint distribution plots.")
    print(f"Latent traversal plotting took {time.time() - start_time:.2f}s")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Main EDA routine with advanced analytics (Optimized & Fixed)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_enhanced_latent_eda(
        model,
        data_for_encoding: np.ndarray,
        metadata_df: pd.DataFrame,
        year_col_name: str,
        x_coord_col_name: str,
        y_coord_col_name: str,
        device='cuda',
        out_dir='enhanced_latent_eda_out',
        batch_size=1024, # Increased default
        k_clusters=8,
        k_range=range(2, 11), # Reduced default range
        random_seed=42,
        # --- Speed/Config Params ---
        plot_dpi=150,
        # t-SNE params
        run_tsne=True,
        n_tsne_samples=5000,
        tsne_n_iter=500,
        tsne_perplexity=30.0,
        tsne_on_pca_components=50,
        use_opentsne_if_available=True,
        # K-Means params
        kmeans_n_init_main=3, # n_init for the main clustering
        # Cluster evaluation params
        run_clustering_evaluation=True, # Flag to run the potentially slow evaluation step
        kmeans_algo_eval='MiniBatchKMeans', # Faster algo for eval loop
        kmeans_n_init_eval=1, # Very low n_init for eval loop
        eval_metric_sample_size=1000, # Subsample size for silhouette in eval loop
        # Latent Traversal Params (NEW)
        run_latent_traversal_plots=True,
        traversal_sample_size=5000, # Subsample for traversal plots
        traversal_n_dims=5,         # Number of top dims to plot
        traversal_n_pairs=3,        # Number of top pairs to plot
        # Other plot params
        run_dendrogram=True,
        dendrogram_n_samples=2000, # Reduced sample size for dendrogram linkage
        run_similarity_analysis=True,
        run_spatiotemporal_analysis=True
    ):
    """
    Perform comprehensive exploratory analysis of the VAE latent space (Optimized & Fixed).
    """
    print(f"--- Running Enhanced Latent EDA (Optimized & Fixed v2) ---")
    print(f"Parameters: run_clustering_evaluation={run_clustering_evaluation}, kmeans_algo_eval='{kmeans_algo_eval}', eval_metric_sample_size={eval_metric_sample_size}, run_tsne={run_tsne}, n_tsne_samples={n_tsne_samples}, tsne_on_pca_components={tsne_on_pca_components}, run_dendrogram={run_dendrogram}, dendrogram_n_samples={dendrogram_n_samples}, run_latent_traversal_plots={run_latent_traversal_plots}, traversal_sample_size={traversal_sample_size}")
    func_start_time = time.time()
    # Set seeds
    np.random.seed(random_seed); torch.manual_seed(random_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_stamped = f"{out_dir}_{timestamp}"
    os.makedirs(out_dir_stamped, exist_ok=True)
    print(f"Output directory: {out_dir_stamped}")

    _device = torch.device(device)
    model = model.to(_device).eval()

    print("Preparing dataset for encoding...")
    if data_for_encoding.dtype != np.float32: data_for_encoding = data_for_encoding.astype(np.float32)
    pytorch_dataset = TensorDataset(torch.from_numpy(data_for_encoding))
    num_loader_workers = 0
    if _device.type != 'cuda' and hasattr(os, 'sched_getaffinity'):
        try: # Handle potential NotImplementedError on some platforms
            affinity_set = os.sched_getaffinity(0)
            if affinity_set: num_loader_workers = len(affinity_set) // 2
        except NotImplementedError:
             num_loader_workers = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 2 else 0

    loader = DataLoader(pytorch_dataset, batch_size=batch_size, shuffle=False, num_workers=num_loader_workers, pin_memory=(_device.type == 'cuda'))

    print(f"Encoding {data_for_encoding.shape[0]} samples...")
    start_time = time.time()
    mu_list, logvar_list = [], []
    with torch.no_grad():
        for i, (x_batch_tuple,) in enumerate(loader):
            x_batch = x_batch_tuple.to(_device)
            mu_batch, logvar_batch = model.encode(x_batch) # ASSUMES (mu, logvar) output
            mu_list.append(mu_batch.cpu()); logvar_list.append(logvar_batch.cpu())
    mu_mat = torch.cat(mu_list).numpy(); logvar_mat = torch.cat(logvar_list).numpy()
    print(f"Encoding completed in {time.time() - start_time:.2f}s.")

    # Create Metadata DataFrame - ensure alignment
    if metadata_df.shape[0] != mu_mat.shape[0]:
         print(f"Warning: Metadata rows ({metadata_df.shape[0]}) != Encoded rows ({mu_mat.shape[0]}). Attempting to use first {mu_mat.shape[0]} metadata rows.")
         meta_df_internal = metadata_df.iloc[:mu_mat.shape[0]].copy()
    else:
         meta_df_internal = metadata_df.copy()
    meta_df_internal['sample_id'] = meta_df_internal.index

    print("Creating DataFrame with latent vectors and metadata...")
    z_mu_cols = [f'z_mu_{d:03d}' for d in range(mu_mat.shape[1])]; z_logvar_cols = [f'z_logvar_{d:03d}' for d in range(mu_mat.shape[1])]
    df_lat = pd.concat([meta_df_internal.reset_index(drop=True), pd.DataFrame(mu_mat, columns=z_mu_cols), pd.DataFrame(logvar_mat, columns=z_logvar_cols)], axis=1)

    print("Analyzing latent space statistics (from mu)..."); start_time = time.time()
    stats_df, corr_mat = analyze_latent_statistics(mu_mat)
    stats_df.to_csv(f'{out_dir_stamped}/latent_dimension_statistics.csv'); np.save(f'{out_dir_stamped}/dimension_correlation_matrix.npy', corr_mat)
    print(f"Statistics calculation took {time.time() - start_time:.2f}s")

    print(f"Performing K-means clustering (k={k_clusters}, n_init={kmeans_n_init_main})..."); start_time = time.time()
    actual_k_clusters = min(k_clusters, mu_mat.shape[0]); labels = np.zeros(mu_mat.shape[0], dtype=int); kmeans_model = None
    if actual_k_clusters > 1:
        kmeans_model = KMeans(n_clusters=actual_k_clusters, n_init=kmeans_n_init_main, random_state=random_seed, algorithm='lloyd')
        labels = kmeans_model.fit_predict(mu_mat)
    else: print(f"Skipping K-means clustering as k_clusters ({actual_k_clusters}) is <= 1.")
    print(f"Main K-Means took {time.time() - start_time:.2f}s")

    cluster_metrics = pd.DataFrame() # Initialize empty
    if run_clustering_evaluation:
        cluster_metrics = evaluate_clustering_quality( mu_mat, labels, k_range=k_range, kmeans_algo_eval=kmeans_algo_eval, kmeans_n_init_eval=kmeans_n_init_eval, metric_subsample_size=eval_metric_sample_size, random_state=random_seed)
        cluster_metrics.to_csv(f'{out_dir_stamped}/cluster_metrics.csv')
    else: print("Skipping clustering quality evaluation.")

    print("Generating cluster profiles..."); start_time = time.time()
    cluster_profiles = generate_cluster_profiles(mu_mat, logvar_mat, labels, meta_df_internal, year_col_name, x_coord_col_name, y_coord_col_name)
    def _np_serializer(obj):
         if isinstance(obj, np.ndarray): return obj.tolist()
         if isinstance(obj, (np.integer, np.floating)): return obj.item()
         if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
         try: return str(obj)
         except Exception: raise TypeError(f"Type {type(obj)} not serializable for cluster profiles")
    try:
        with open(f'{out_dir_stamped}/cluster_profiles.json', 'w') as f: json.dump(cluster_profiles, f, indent=2, default=_np_serializer)
    except TypeError as e: print(f"Error serializing cluster profiles to JSON: {e}. Profiles might contain unsupported types.")
    print(f"Cluster profile generation took {time.time() - start_time:.2f}s")

    print("Generating KL divergence analysis..."); start_time = time.time()
    kl_divergence_values = -0.5 * np.sum(1 + logvar_mat - mu_mat**2 - np.exp(logvar_mat), axis=1)
    plot_kl_analysis(kl_divergence_values, out_dir_stamped, plot_dpi=plot_dpi)
    print(f"KL Analysis plotting took {time.time() - start_time:.2f}s")

    print("Running PCA and creating visualizations..."); start_time = time.time()
    pc_vectors, pca_model_fitted = plot_pca_analysis(mu_mat, meta_df_internal, labels, year_col_name, out_dir_stamped, plot_dpi=plot_dpi)
    print(f"PCA + Plotting took {time.time() - start_time:.2f}s")

    if run_clustering_evaluation or cluster_profiles:
        print("Creating cluster analysis visualizations..."); start_time = time.time()
        plot_clustering_analysis(mu_mat, labels, out_dir_stamped, cluster_metrics, cluster_profiles, plot_dpi=plot_dpi, dendrogram_n_samples=dendrogram_n_samples if run_dendrogram else 0)
        print(f"Cluster analysis plotting took {time.time() - start_time:.2f}s")

    if run_similarity_analysis:
        print("Generating similarity analysis and dimension statistics..."); start_time = time.time()
        plot_advanced_similarity_analysis(mu_mat, labels, stats_df, corr_mat, out_dir_stamped, plot_dpi=plot_dpi)
        print(f"Similarity/Stats plotting took {time.time() - start_time:.2f}s")

    if run_spatiotemporal_analysis:
        print("Analyzing spatial and temporal patterns..."); start_time = time.time()
        spatiotemporal_df = plot_spatial_year_analysis(mu_mat, logvar_mat, meta_df_internal, labels, pc_vectors, year_col_name, x_coord_col_name, y_coord_col_name, out_dir_stamped, plot_dpi=plot_dpi)
        print(f"Spatial/Temporal plotting took {time.time() - start_time:.2f}s")

    # --- Optimized Latent Traversal Call ---
    if run_latent_traversal_plots:
        plot_latent_traversal(
            mu_mat, stats_df, out_dir_stamped,
            plot_dpi=plot_dpi,
            traversal_sample_size=traversal_sample_size, # Pass param
            n_dims_traversal=traversal_n_dims,           # Pass param
            n_pairs_traversal=traversal_n_pairs          # Pass param
        )
    else: print("Skipping latent traversal plots as per configuration.")

    # --- Optimized t-SNE Section ---
    tsne_embedding_final = None; tsne_df_final = None
    if run_tsne:
        print("Preparing for t-SNE..."); start_time_tsne = time.time()
        data_for_tsne = mu_mat; indices_tsne = np.arange(mu_mat.shape[0])
        if mu_mat.shape[0] > n_tsne_samples:
            print(f"Subsampling {n_tsne_samples} from {mu_mat.shape[0]} for t-SNE.")
            indices_tsne = np.random.choice(mu_mat.shape[0], n_tsne_samples, replace=False)
            data_for_tsne = mu_mat[indices_tsne, :]

        tsne_input_dim = data_for_tsne.shape[1]; tsne_input_source = "mu_mat"
        if tsne_on_pca_components is not None and pc_vectors is not None and pc_vectors.shape[0] == mu_mat.shape[0]:
             effective_pca_comps = min(tsne_on_pca_components, pc_vectors.shape[1])
             if effective_pca_comps > 1:
                  print(f"Using top {effective_pca_comps} PCs for t-SNE.")
                  data_for_tsne = pc_vectors[indices_tsne, :effective_pca_comps]
                  tsne_input_dim = effective_pca_comps; tsne_input_source = f"{effective_pca_comps} PCs"
             else: print(f"Not enough PCA components ({pc_vectors.shape[1]}), using original data for t-SNE.")
        elif tsne_on_pca_components is not None: print("PCA vectors not available, using original data for t-SNE.")

        current_n_samples_for_tsne = data_for_tsne.shape[0]
        actual_perplexity = min(tsne_perplexity, current_n_samples_for_tsne - 1)
        if actual_perplexity < 5 and current_n_samples_for_tsne >= 5: actual_perplexity = 5

        if current_n_samples_for_tsne > 1 and actual_perplexity >= 1:
            tsne_kwargs = {"n_components": 2, "perplexity": actual_perplexity, "random_state": random_seed, "n_jobs": -1}
            tsne_impl_name = "scikit-learn TSNE"
            if use_opentsne_if_available and OPEN_TSNE_AVAILABLE:
                tsne_kwargs["early_exaggeration_iter"] = max(50, tsne_n_iter // 4); tsne_kwargs["n_iter"] = max(100, tsne_n_iter * 3 // 4)
                if "n_jobs" in tsne_kwargs: del tsne_kwargs["n_jobs"]
                tsne_model = OpenTSNE_TSNE(**tsne_kwargs); tsne_impl_name = "OpenTSNE"
                print(f"Running {tsne_impl_name} (Perp={actual_perplexity:.1f}, NIter~{tsne_n_iter}) on {current_n_samples_for_tsne} samples ({tsne_input_source})...")
                tsne_embedding_final = tsne_model.fit(data_for_tsne.astype(np.float64)) # OpenTSNE uses .fit()
            else:
                tsne_kwargs["init"] = 'pca'; tsne_kwargs["learning_rate"] = 'auto'; tsne_kwargs["n_iter"] = tsne_n_iter
                tsne_model = SKL_TSNE(**tsne_kwargs)
                print(f"Running {tsne_impl_name} (Perp={actual_perplexity:.1f}, NIter={tsne_n_iter}) on {current_n_samples_for_tsne} samples ({tsne_input_source})...")
                tsne_embedding_final = tsne_model.fit_transform(data_for_tsne.astype(np.float64)) # Sklearn uses .fit_transform()

            print(f"t-SNE calculation took {time.time() - start_time_tsne:.2f}s.")

            # Create t-SNE DataFrame and plots
            tsne_meta_input = meta_df_internal.iloc[indices_tsne].reset_index(drop=True)
            tsne_labels_input = labels[indices_tsne] if labels is not None else np.zeros(len(indices_tsne))
            tsne_df_data = {'TSNE1': tsne_embedding_final[:, 0], 'TSNE2': tsne_embedding_final[:, 1], 'cluster': tsne_labels_input}
            if year_col_name and year_col_name in tsne_meta_input.columns: tsne_df_data[year_col_name] = tsne_meta_input[year_col_name]
            if x_coord_col_name and x_coord_col_name in tsne_meta_input.columns: tsne_df_data['grid_x'] = tsne_meta_input[x_coord_col_name]
            if y_coord_col_name and y_coord_col_name in tsne_meta_input.columns: tsne_df_data['grid_y'] = tsne_meta_input[y_coord_col_name]
            tsne_df_final = pd.DataFrame(tsne_df_data)
            plt.figure(figsize=(10, 8)); sns.scatterplot(x='TSNE1', y='TSNE2', hue='cluster', data=tsne_df_final, palette='tab10', alpha=0.7, s=8, legend='auto'); plt.title(f'{tsne_impl_name} (Perp {actual_perplexity:.1f})'); plt.xlabel('t-SNE Dim 1'); plt.ylabel('t-SNE Dim 2'); plt.tight_layout(); plt.savefig(f'{out_dir_stamped}/tsne.png', dpi=plot_dpi); plt.close()
            fig_tsne_plotly = px.scatter(tsne_df_final, x='TSNE1', y='TSNE2', color='cluster', symbol=year_col_name if year_col_name and year_col_name in tsne_df_final.columns else None, hover_data=[col for col in ['grid_x', 'grid_y', year_col_name] if col in tsne_df_final.columns], title='t-SNE (Interactive)', color_discrete_map={-1: "lightgrey"}); fig_tsne_plotly.write_html(f'{out_dir_stamped}/tsne_interactive.html')
        else: print(f"Skipping t-SNE plot due to insufficient samples/perplexity.")
    else: print("Skipping t-SNE as per configuration.")

    print("Saving final artifacts..."); start_time = time.time()
    df_lat['cluster'] = labels; df_lat['kl_divergence'] = kl_divergence_values
    df_lat.to_parquet(f'{out_dir_stamped}/latent_vectors_with_metadata.parquet')
    projection_data_dict = {
        'pc_vectors': pc_vectors, 'tsne_embedding_sampled': tsne_embedding_final,
        'pca_explained_variance': pca_model_fitted.explained_variance_ratio_ if pca_model_fitted else None,
        'cluster_centers_kmeans': kmeans_model.cluster_centers_ if kmeans_model and hasattr(kmeans_model, 'cluster_centers_') else None
    }
    with open(f'{out_dir_stamped}/projection_data.pkl', 'wb') as f: pickle.dump(projection_data_dict, f)

    # Generate summary report (fast)
    generate_analysis_summary(model, mu_mat, logvar_mat, stats_df, cluster_profiles, cluster_metrics, out_dir_stamped)
    print(f"Artifact saving took {time.time() - start_time:.2f}s")

    print(f'--- Enhanced EDA Completed in {time.time() - func_start_time:.2f} seconds. Results saved to "{out_dir_stamped}/" ---')
    return df_lat, stats_df, cluster_profiles


# --- Summary Report Generation (Unchanged from previous fix) ---
def generate_analysis_summary(model, mu_mat, logvar_mat, stats_df, cluster_profiles, cluster_metrics, out_dir):
     """Generate a comprehensive analysis summary markdown report"""
     latent_dim = mu_mat.shape[1]; n_samples = mu_mat.shape[0]; kl_per_sample = -0.5 * np.sum(1 + logvar_mat - mu_mat**2 - np.exp(logvar_mat), axis=1); avg_kl_total = np.mean(kl_per_sample); active_dims = 0
     if 'std' in stats_df.columns: active_dims = np.sum(stats_df['std'] > 0.1)
     best_k_silhouette = 'N/A'
     if not cluster_metrics.empty and 'silhouette' in cluster_metrics.columns and cluster_metrics['silhouette'].notna().any():
          best_k_row_idx = cluster_metrics['silhouette'].dropna().idxmax()
          if pd.notna(best_k_row_idx): best_k_row = cluster_metrics.loc[best_k_row_idx]; best_k_silhouette = int(best_k_row['k']) if pd.notna(best_k_row['k']) else 'N/A'
     report = f"""# Latent Space Analysis Report\n\n## Overview\n- **Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n- **Latent Dimension:** {latent_dim}\n- **Number of Samples:** {n_samples:,}\n- **VAE Model Type:** {type(model).__name__}\n- **Average KL Divergence (Total):** {avg_kl_total:.4f}\n- **Active Dimensions (std > 0.1):** {active_dims}\n- **Optimal Cluster Count (Silhouette):** {best_k_silhouette}\n\n## Latent Space Statistics (based on mu)\n"""
     if not stats_df.empty and 'importance_score' in stats_df.columns:
          top_dims_summary = stats_df.sort_values('importance_score', ascending=False).head(10)
          report += """\n### Top 10 Important Dimensions (mu-space):\n| Dimension | Importance | Mean   | Std Dev | Skewness | Kurtosis | Activation |\n|-----------|------------|--------|---------|----------|----------|------------|\n"""
          for dim_name, row_data in top_dims_summary.iterrows(): report += f"| {dim_name} | {row_data['importance_score']:.2f} | {row_data['mean']:.2f} | {row_data['std']:.2f} | {row_data['skewness']:.2f} | {row_data['kurtosis']:.2f} | {row_data['activation_ratio']:.2f} |\n"
     else: report += "\nLatent dimension statistics (importance score) not available.\n"
     if cluster_profiles:
          report += """\n### Cluster Analysis Summary:\n| Cluster ID | Size   | Proportion | Avg KL | Dispersion (mu) |\n|------------|--------|------------|--------|-----------------|\n"""
          for profile in cluster_profiles: report += f"| {profile['cluster_id']} | {profile['size']:,} | {profile['proportion']:.3f} | {profile.get('avg_kl', np.nan):.2f} | {profile.get('dispersion_mu', np.nan):.2f} |\n"
     else: report += "\nCluster profiles not available.\n"
     report += """\n## Interpretation and Insights (General)\nThe KL divergence, PCA, t-SNE, and clustering results provide a multi-faceted view of the learned latent space.\n- **KL Divergence**: Indicates how much each encoded point deviates from the prior. Higher KL samples might be outliers or represent more complex data.\n- **PCA**: Shows principal directions of variance in the latent space.\n- **t-SNE/UMAP**: Offers non-linear dimensionality reduction for visualizing clusters and local structure.\n- **Clustering**: Identifies distinct groups of samples in the latent space, which may correspond to different data archetypes.\n- **Spatial/Temporal Analysis**: Relates latent representations back to their original spatial or temporal context.\n\nFurther investigation should correlate these latent features and clusters with known characteristics of the input data.\n"""
     with open(f'{out_dir}/analysis_report.md', 'w') as f: f.write(report)
     viz_index = """# Visualization Index
## Dimension Analysis
- [Dimension Statistics](dimension_statistics.png)
- [Top Dimensions Distribution (Hist)](top_dimensions_distribution_hist.png)
- [Top Dimensions Joint Distribution (Hist)](top_dimensions_joint_hist.png)
- [Top Dimensions Correlation](top_dims_correlation.png)
## KL Divergence Analysis
- [KL Analysis](kl_analysis.png)
- [KL by Year](kl_by_year.png)
## Clustering Analysis
- [Cluster Evaluation](cluster_evaluation.png)
- [Cluster Profiles](cluster_profiles.png)
- [Dendrogram](dendrogram.png)
## Projections
- [PCA Variance](pca_variance.png)
- [PCA Loadings](pca_loadings.png)
- [t-SNE](tsne.png)
## Spatial-Temporal Analysis
- [Spatial Clusters](spatial_clusters.png)
- [Temporal Clusters](temporal_clusters.png)
## Interactive Visualizations
- [PCA Interactive](pca_interactive.html)
- [PCA 3D](pca_3d.html)
- [t-SNE Interactive](tsne_interactive.html)
- [Spatial Interactive](spatial_interactive.html)
""" # Add UMAP links if enabled
     with open(f'{out_dir}/visualization_index.md', 'w') as f: f.write(viz_index)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# EXAMPLE USAGE WRAPPER (Updated with speed parameters)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def analyze_vae_latent_space_enhanced(
        model,
        data_for_encoding: np.ndarray,
        metadata_df: pd.DataFrame,
        year_col_name: str,
        x_coord_col_name: str,
        y_coord_col_name: str,
        output_folder="enhanced_latent_eda_results",
        device="cuda",
        batch_size=1024, # Match run_enhanced_latent_eda default
        k_clusters_config=8,
        k_range_config=range(2, 11), # Smaller default range
        seed=42,
        # --- Pass Speed/Config Params ---
        eda_plot_dpi=150,
        eda_run_tsne=True,
        eda_n_tsne_samples=5000,
        eda_tsne_n_iter=500,
        eda_tsne_perplexity=30.0,
        eda_tsne_on_pca_components=50,
        eda_use_opentsne=True,
        eda_kmeans_n_init_main=3,
        eda_run_clustering_evaluation=True,
        eda_kmeans_algo_eval='MiniBatchKMeans',
        eda_kmeans_n_init_eval=1,
        eda_eval_metric_sample_size=1000,
        eda_run_dendrogram=True,
        eda_dendrogram_n_samples=2000,
        eda_run_similarity_analysis=True,
        eda_run_spatiotemporal_analysis=True,
        # Latent Traversal Params (NEW)
        eda_run_latent_traversal_plots=True,
        eda_traversal_sample_size=5000,
        eda_traversal_n_dims=5,
        eda_traversal_n_pairs=3
    ):
    """Convenience function to run the complete enhanced latent space analysis (Optimized & Fixed)."""
    print(f"Starting Enhanced VAE Latent Space Analysis (Optimized). Output base: {output_folder}")

    # Make sure the main function (run_enhanced_latent_eda) is defined/imported
    df_latent_vectors, df_stats, list_cluster_profiles = run_enhanced_latent_eda(
        model=model, data_for_encoding=data_for_encoding, metadata_df=metadata_df,
        year_col_name=year_col_name, x_coord_col_name=x_coord_col_name, y_coord_col_name=y_coord_col_name,
        device=device, out_dir=output_folder, batch_size=batch_size,
        k_clusters=k_clusters_config, k_range=k_range_config, random_seed=seed,
        # Pass speed/config params
        plot_dpi=eda_plot_dpi, run_tsne=eda_run_tsne, n_tsne_samples=eda_n_tsne_samples,
        tsne_n_iter=eda_tsne_n_iter, tsne_perplexity=eda_tsne_perplexity,
        tsne_on_pca_components=eda_tsne_on_pca_components, use_opentsne_if_available=eda_use_opentsne,
        kmeans_n_init_main=eda_kmeans_n_init_main, run_clustering_evaluation=eda_run_clustering_evaluation,
        kmeans_algo_eval=eda_kmeans_algo_eval, kmeans_n_init_eval=eda_kmeans_n_init_eval,
        eval_metric_sample_size=eda_eval_metric_sample_size, run_dendrogram=eda_run_dendrogram,
        dendrogram_n_samples=eda_dendrogram_n_samples, run_similarity_analysis=eda_run_similarity_analysis,
        run_spatiotemporal_analysis=eda_run_spatiotemporal_analysis,
        # Pass new traversal params
        run_latent_traversal_plots=eda_run_latent_traversal_plots,
        traversal_sample_size=eda_traversal_sample_size,
        traversal_n_dims=eda_traversal_n_dims,
        traversal_n_pairs=eda_traversal_n_pairs
    )

    print(f"Enhanced analysis complete! Results saved to a timestamped subfolder within '{output_folder}'.")
    if df_latent_vectors is not None and not df_latent_vectors.empty:
        print("\nKey insights from enhanced EDA:")
        print(f"- Number of samples analyzed: {len(df_latent_vectors):,}")
        if df_stats is not None and not df_stats.empty:
            print(f"- Latent dimensions: {df_stats.shape[0]}")
            if 'importance_score' in df_stats.columns:
                 print(f"- Top 3 most important dimensions (mu-space): {', '.join(df_stats.sort_values('importance_score', ascending=False).index[:3])}")
        if list_cluster_profiles:
            print(f"- Clusters found: {len(list_cluster_profiles)}")
            if list_cluster_profiles:
                top_cluster_profile = sorted(list_cluster_profiles, key=lambda x: x.get('size', 0), reverse=True)[0]
                if top_cluster_profile:
                    print(f"- Largest cluster (ID {top_cluster_profile.get('cluster_id')}) contains {top_cluster_profile.get('size', 0):,} samples ({top_cluster_profile.get('proportion', 0):.1%} of data)")
            else: print("- No valid cluster profiles generated.")

    return df_latent_vectors

