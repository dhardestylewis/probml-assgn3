# -*- coding: utf-8 -*-
from probml.core.utils import get_logger

# File: validation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, BaseCrossValidator
from typing import Tuple, Optional, Iterator, Union
import logging


def get_time_series_splitter(
    df: pd.DataFrame,
    date_col: str,
    n_splits: int = 5,
    max_train_size: Optional[int] = None,
    test_size: Optional[int] = None, # Added optional test_size
    gap: int = 0                 # Added optional gap (in terms of samples)
) -> Tuple[TimeSeriesSplit, np.ndarray]:
    """
    Builds and returns an sklearn TimeSeriesSplit instance configured for the
    input DataFrame, along with the DataFrame's index sorted by the date column.

    Args:
        df: The full DataFrame (must include date_col).
        date_col: The name of the datetime-like column to sort by.
        n_splits: The number of splitting iterations in the cross-validator.
        max_train_size: Maximum size for a single training set. None means no limit.
        test_size: Used to limit the size of the test set. None means it's determined by n_splits.
        gap: Number of samples to exclude between the end of the train set and
             the start of the test set.

    Returns:
        A tuple containing:
            - tscv: An initialized TimeSeriesSplit instance.
            - ordered_idx: A NumPy array of the original DataFrame index, sorted by date_col.

    Raises:
        ValueError: If date_col is not found in df or cannot be converted to datetime.
        ValueError: If df is empty after handling NaNs in date_col.

    Example:
        tscv, ordered_idx = get_time_series_splitter(df, 'sale_date', n_splits=4)
        # Example usage in a training loop:
        for train_indices, test_indices in tscv.split(ordered_idx):
            # Use the *original* index values from ordered_idx
            train_original_idx = ordered_idx[train_indices]
            test_original_idx = ordered_idx[test_indices]
            # Select data from the original DataFrame using these indices
            train_df = df.loc[train_original_idx]
            test_df  = df.loc[test_original_idx]
            # ... proceed with training/evaluation ...
    """
    logger.info(f"Configuring TimeSeriesSplit (n_splits={n_splits}, date_col='{date_col}', max_train={max_train_size}, test_size={test_size}, gap={gap})")

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")

    # Attempt to convert to datetime and sort, handling potential errors
    try:
        # Work on a copy to avoid modifying original df outside function scope
        df_sorted = df[[date_col]].copy()
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors='coerce')

        # Handle NaNs in the date column before sorting
        initial_rows = len(df_sorted)
        df_sorted.dropna(subset=[date_col], inplace=True)
        if len(df_sorted) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(df_sorted)} rows with invalid/NaN dates in '{date_col}'.")

        if df_sorted.empty:
             raise ValueError(f"DataFrame is empty after dropping rows with invalid dates in '{date_col}'.")

        df_sorted = df_sorted.sort_values(by=date_col, ascending=True)
        ordered_idx = df_sorted.index.to_numpy() # Get the original index values in sorted order

    except Exception as e:
        logger.error(f"Error processing date column '{date_col}': {e}", exc_info=True)
        raise ValueError(f"Could not process date column '{date_col}'. Ensure it's datetime-like.") from e

    # Instantiate TimeSeriesSplit
    # Note: TimeSeriesSplit uses the *number* of samples, not dates directly
    try:
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=max_train_size,
            test_size=test_size,
            gap=gap
        )
        # Perform a dummy split to check if parameters are valid for the data size
        _ = next(tscv.split(ordered_idx))
    except ValueError as e_tscv:
         logger.error(f"Error initializing TimeSeriesSplit with given parameters for data size {len(ordered_idx)}: {e_tscv}", exc_info=True)
         raise ValueError(f"TimeSeriesSplit parameters invalid for data size {len(ordered_idx)}: {e_tscv}") from e_tscv

    logger.info(f"TimeSeriesSplit configured. Returning splitter and sorted index array (length {len(ordered_idx)}).")
    return tscv, ordered_idx


def expanding_window_split(
    df: pd.DataFrame,
    date_col: str,
    initial_train_period: pd.Timedelta,
    test_period: pd.Timedelta,
    gap: pd.Timedelta = pd.Timedelta(days=0), # Ensure default is Timedelta
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates integer position indices for expanding-window time series splits.

    The training window starts with `initial_train_period` and expands by `test_period`
    in each subsequent split. A 'gap' can be introduced between the train and test sets.

    Args:
        df: The full DataFrame (must include date_col).
        date_col: The name of the datetime-like column to determine splits.
        initial_train_period: The duration of the initial training set.
        test_period: The duration of each test set (and the amount the training
                     window expands by each time).
        gap: The duration of the gap between the end of the training period and
             the start of the test period. Defaults to 0.

    Yields:
        Iterator of tuples, where each tuple contains:
            - train_pos: NumPy array of integer positions (iloc) for the training set
                         within the date-sorted DataFrame.
            - test_pos: NumPy array of integer positions (iloc) for the test set
                        within the date-sorted DataFrame.

    Raises:
        ValueError: If date_col is not found, cannot be converted to datetime,
                    or if periods/gap are invalid.
        ValueError: If df is empty after handling NaNs in date_col.

    Example:
        df_sorted = df.sort_values('sale_date') # Sort beforehand is recommended
        splitter = expanding_window_split(
            df_sorted, 'sale_date',
            initial_train_period=pd.Timedelta('730D'), # 2 years
            test_period=pd.Timedelta('180D'),        # 6 months
            gap=pd.Timedelta('30D')                 # 1 month gap
        )
        for train_positions, test_positions in splitter:
            train_subset = df_sorted.iloc[train_positions]
            test_subset  = df_sorted.iloc[test_positions]
            # ... train model on train_subset, evaluate on test_subset ...
    """
    logger.info(f"Configuring Expanding Window Split (date_col='{date_col}', initial='{initial_train_period}', "
                f"test='{test_period}', gap='{gap}')")

    # --- Input Validation ---
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
    if not isinstance(initial_train_period, pd.Timedelta) or initial_train_period <= pd.Timedelta(0):
         raise ValueError("initial_train_period must be a positive pd.Timedelta.")
    if not isinstance(test_period, pd.Timedelta) or test_period <= pd.Timedelta(0):
         raise ValueError("test_period must be a positive pd.Timedelta.")
    if not isinstance(gap, pd.Timedelta) or gap < pd.Timedelta(0):
         raise ValueError("gap must be a non-negative pd.Timedelta.")

    # --- Data Preparation ---
    try:
        # Select only necessary columns and work on a copy
        df_sorted = df[[date_col]].copy()
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors='coerce')

        initial_rows = len(df_sorted)
        df_sorted.dropna(subset=[date_col], inplace=True)
        if len(df_sorted) < initial_rows:
            logger.warning(f"ExpandingWindow: Dropped {initial_rows - len(df_sorted)} rows with invalid/NaN dates in '{date_col}'.")

        if df_sorted.empty:
             raise ValueError(f"ExpandingWindow: DataFrame is empty after dropping rows with invalid dates in '{date_col}'.")

        df_sorted = df_sorted.sort_values(by=date_col, ascending=True)
        # Get the date series for efficient comparison
        dates_sorted = df_sorted[date_col]
        n_samples = len(dates_sorted)

    except Exception as e:
        logger.error(f"Error processing date column '{date_col}' for expanding window: {e}", exc_info=True)
        raise ValueError(f"Could not process date column '{date_col}'. Ensure it's datetime-like.") from e

    # --- Splitting Logic ---
    t_min = dates_sorted.iloc[0]
    t_max = dates_sorted.iloc[-1]
    k = 0
    split_num = 1

    while True:
        # Calculate time boundaries for the current split
        current_train_end = t_min + initial_train_period + (k * test_period)
        current_test_start = current_train_end + gap
        current_test_end = current_test_start + test_period

        # Check if the test period goes beyond the available data
        if current_test_start >= t_max:
            logger.info(f"Stopping expanding window: Test start ({current_test_start}) >= max data date ({t_max}). Generated {split_num-1} splits.")
            break

        # Find integer positions using boolean indexing (often efficient enough)
        # Alternatively, use np.searchsorted for potentially better performance on very large data
        train_mask = (dates_sorted < current_train_end)
        test_mask = (dates_sorted >= current_test_start) & (dates_sorted < current_test_end)

        train_pos = np.where(train_mask)[0]
        test_pos = np.where(test_mask)[0]

        # Check if the test set is empty for this iteration
        if len(test_pos) == 0:
            # Check if the train window has already covered all data
            if current_train_end >= t_max:
                 logger.info(f"Stopping expanding window: Train end ({current_train_end}) covers all data. Test set empty. Generated {split_num-1} splits.")
            else:
                 logger.info(f"Stopping expanding window: Test set for period [{current_test_start}, {current_test_end}) is empty. Generated {split_num-1} splits.")
            break

        # Ensure train set is not empty (could happen with very small initial period)
        if len(train_pos) == 0:
            logger.warning(f"Split {split_num}: Training set is empty (train_end={current_train_end}). Check initial_train_period. Skipping this split.")
            k += 1 # Move to the next potential window
            continue


        logger.debug(f"Split {split_num}: Train Period [< {current_train_end}], Test Period [{current_test_start}, {current_test_end}) -> Train size: {len(train_pos)}, Test size: {len(test_pos)}")
        yield train_pos, test_pos

        k += 1
        split_num += 1

    if split_num == 1: # Check if the loop never yielded anything
        logger.warning("Expanding window did not generate any valid splits. Check time periods relative to data range.")

