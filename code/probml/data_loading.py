# -*- coding: utf-8 -*-
from probml.core.utils import get_logger


# Example of how to call this from the main pipeline (replace placeholders)
# if __name__ == "__main__":
#      # Assuming main_pipeline_vae_model, main_pipeline_vae_input_data, etc. are defined
#      # And assuming the new speed/config parameters are defined in the main pipeline script, e.g.:
#      # EDA_ENHANCED_CFG = { 'eda_plot_dpi': 150, 'eda_run_tsne': True, ... etc }
#      if 'main_pipeline_vae_model' in locals():
#          df_result_enhanced_opt = analyze_vae_latent_space_enhanced(
#               model=main_pipeline_vae_model, data_for_encoding=main_pipeline_vae_input_data,
#               metadata_df=main_pipeline_df_for_metadata, year_col_name=main_pipeline_year_col,
#               x_coord_col_name=main_pipeline_x_col, y_coord_col_name=main_pipeline_y_col,
#               device=main_pipeline_device, batch_size=main_pipeline_vae_batch_size,
#               # Pass configured parameters
#               **EDA_ENHANCED_CFG # Pass the dictionary of speed parameters
#          )

PARQUET_PATH = '/content/drive/MyDrive/e6691_2025Spring_nyre_local/data/processed_building_ml.parquet'

# --- Data Loading Function ---
def load_nyc_data(file_path='building_ml_merged.csv', logger_instance=None):
    """Loads NYC data from CSV or Parquet."""
    log = logger_instance or get_logger("LoadNYCData", verbose=True)
    try:
        log.info(f"Attempting to load data from: {file_path}")
        if not os.path.exists(file_path):
            log.error(f"File not found: {file_path}")
            return None
        # Determine file type and load
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path, low_memory=False)
        elif file_path.lower().endswith(('.parquet', '.parq')):
            df = pd.read_parquet(file_path)
        else:
            log.error(f"Unsupported file format: {file_path}.")
            return None
        log.info(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        log.error(f"Error loading data from {file_path}: {e}", exc_info=True)
        return None

