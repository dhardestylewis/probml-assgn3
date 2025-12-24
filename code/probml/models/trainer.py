# -*- coding: utf-8 -*-
from probml.core.utils import get_logger
from probml.models.vae_base import VariationalAutoencoderBase
from probml.models.miwae import SemiSupMIWAE

# (VAETrainer, prepare_vae_input, create_vae_from_artifacts, and helper functions will follow)
# ... (Rest of the VAE Pipeline file starting from VAETrainer)

# Assuming necessary imports are available from other modules or defined above:
# import torch, torch.nn as nn, torch.optim as optim, etc.
# from vae_base_module import VariationalAutoencoderBase, SemiSupMIWAE # Example import
# from logger_setup import get_logger # Example import
# from distributions import StudentTDistribution, MixtureDistribution # Example import

import math
import os
import pickle
import time
import logging
from typing import List, Optional, Dict, Tuple, Union, Any, NamedTuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Import necessary distributions
from torch.distributions import (
    Normal, StudentT, Categorical, MixtureSameFamily, Independent,
    MultivariateNormal, Distribution, utils as dist_utils
)
# Example Scaler/Model imports (adjust paths as needed)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- VAE Trainer Class (Refactored for Semi-Supervised MIWAE) ---
class VAETrainer:
    """
    Handles training, evaluation, imputation, saving/loading of VAE/SemiSupMIWAE models.
    Manages loss calculation including reconstruction for X, KLD, and optional supervised loss for Y (target).
    """
    def __init__(self,
                 model: Union[VariationalAutoencoderBase, SemiSupMIWAE],
                 learning_rate: float = 1e-3,
                 kld_weight: float = 1.0, # Base KLD weight (beta), can be annealed
                 alpha_price_loss: Optional[float] = None, # Weight for supervised Y loss, trainer can override model's alpha
                 optimizer_str: str = 'adam',
                 reconstruction_loss_type_x: str = 'mse', # Loss type for X features: 'mse' or 'huber'
                 price_loss_type_y: str = 'gaussian_nll', # Loss type for Y target: 'gaussian_nll' or 'mse'
                 huber_delta: float = 1.0, # Delta for Huber loss (if used for X)
                 device: Optional[torch.device] = None,
                 results_dir: str = "vae_results",
                 verbose: bool = True):

        self.model = model
        self.is_semi_supervised = isinstance(model, SemiSupMIWAE)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = results_dir
        self.verbose = verbose
        self.kld_weight = kld_weight
        self.logger = get_logger(self.__class__.__name__, verbose)

        # Determine alpha_price_loss for the Y target
        if self.is_semi_supervised:
            model_alpha = getattr(model, 'alpha_price_loss', 1.0) # Default from model if exists
            if alpha_price_loss is not None: # Trainer override
                self.alpha_price_loss = alpha_price_loss
                if abs(self.alpha_price_loss - model_alpha) > 1e-6: # Log if different
                     self.logger.info(f"Trainer overriding model's alpha_price_loss ({model_alpha}) with: {self.alpha_price_loss}")
            else: # Use model's alpha if trainer doesn't override
                self.alpha_price_loss = model_alpha
        else: # Unsupervised case
            self.alpha_price_loss = 0.0

        self.reconstruction_loss_type_x = reconstruction_loss_type_x.lower()
        self.price_loss_type_y = price_loss_type_y.lower() if self.is_semi_supervised else 'none'
        self.huber_delta = huber_delta

        os.makedirs(self.results_dir, exist_ok=True)
        self.model.to(self.device)

        # Setup Optimizer
        optimizer_map = {'adam': optim.Adam, 'adamw': optim.AdamW, 'rmsprop': optim.RMSprop}
        optimizer_class = optimizer_map.get(optimizer_str.lower(), optim.Adam)
        if optimizer_str.lower() not in optimizer_map:
            self.logger.warning(f"Unknown optimizer '{optimizer_str}'. Defaulting to Adam.")
        self.optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)

        self.history: Dict[str, List[float]] = defaultdict(list) # Stores training history

        model_type_str = self.model.__class__.__name__
        self._log(f"VAETrainer initialized for {model_type_str}. Device: {self.device}. "
                  f"Optimizer: {optimizer_str}. LR: {learning_rate}. "
                  f"ReconLoss(X): '{self.reconstruction_loss_type_x}'. "
                  f"Alpha(Y Loss): {self.alpha_price_loss if self.is_semi_supervised else 'N/A'}. "
                  f"LossType(Y): '{self.price_loss_type_y if self.is_semi_supervised else 'N/A'}'",
                  level="info")

    def _log(self, message: str, level: str = "info", exc_info: bool = False):
        """Helper for conditional logging based on self.verbose"""
        current_log_level_val = getattr(logging, level.upper(), logging.INFO)
        if self.verbose or current_log_level_val >= logging.WARNING:
            log_func = getattr(self.logger, level, self.logger.info)
            log_func(message, exc_info=exc_info)

    def _calculate_y_loss(self, y_true: torch.Tensor, y_pred_mu: torch.Tensor, y_pred_logvar: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the target variable Y, considering only observed values specified by y_mask.
        Supports 'gaussian_nll' or 'mse'.
        """
        # Ensure y_true, y_pred_mu, y_pred_logvar, y_mask have compatible shapes
        # y_true, y_pred_mu, y_pred_logvar likely (batch_size, y_dim)
        # y_mask likely (batch_size, y_dim)

        num_observed_y = y_mask.sum().clamp(min=1) # Avoid division by zero

        if self.price_loss_type_y == 'gaussian_nll':
            y_pred_logvar_clamped = torch.clamp(y_pred_logvar, min=-15.0, max=15.0) # Stability
            variance = torch.exp(y_pred_logvar_clamped) + 1e-9 # Add epsilon
            # Gaussian NLL per element: 0.5 * (log(2*pi) + log_variance + ((y_true - y_pred_mu)^2 / variance))
            elementwise_loss = 0.5 * (math.log(2 * math.pi) + y_pred_logvar_clamped +
                                     F.mse_loss(y_pred_mu, y_true, reduction='none') / variance)
        elif self.price_loss_type_y == 'mse':
            # Simple MSE loss for Y, does not use y_pred_logvar
            elementwise_loss = F.mse_loss(y_pred_mu, y_true, reduction='none')
        else:
            self.logger.error(f"Unknown price_loss_type_y: '{self.price_loss_type_y}'. Add support or use 'gaussian_nll'/'mse'. Defaulting to MSE.")
            elementwise_loss = F.mse_loss(y_pred_mu, y_true, reduction='none')

        # Apply mask (element-wise multiply) and average over observed elements
        masked_loss = elementwise_loss * y_mask
        final_y_loss = masked_loss.sum() / num_observed_y

        # Safety check for NaN/Inf in final Y loss
        if torch.isnan(final_y_loss) or torch.isinf(final_y_loss):
            self.logger.error(f"Loss for Y (target) became NaN/Inf using type '{self.price_loss_type_y}'. "
                              f"Check inputs: y_true range [{y_true.min():.2f},{y_true.max():.2f}], "
                              f"y_pred_mu range [{y_pred_mu.min():.2f},{y_pred_mu.max():.2f}], "
                              f"y_pred_logvar range [{y_pred_logvar.min():.2f},{y_pred_logvar.max():.2f}]. "
                              f"Observed Y count: {num_observed_y.item()}. Setting Y loss to 0 for this batch.")
            return torch.tensor(0.0, device=self.device)

        return final_y_loss

    def _calculate_loss(self,
                        x_batch: torch.Tensor,       # Input features X, (batch, x_dim)
                        x_mask_batch: torch.Tensor,  # Mask for X, (batch, x_dim)
                        y_batch: Optional[torch.Tensor], # Target Y, (batch, y_dim)
                        y_mask_batch: Optional[torch.Tensor],# Mask for Y, (batch, y_dim)
                        model_outputs: Tuple,        # Outputs from model.forward()
                        current_beta_kld: float = 1.0  # Annealing factor for KLD
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates the total loss (ELBO negative) and its components."""

        # Unpack model outputs
        if self.is_semi_supervised:
            recon_x, mu_latent, logvar_latent, y_pred_mu, y_pred_logvar = model_outputs
        else: # Unsupervised
            recon_x, mu_latent, logvar_latent = model_outputs
            y_pred_mu, y_pred_logvar, y_batch, y_mask_batch = None, None, None, None # Ensure Y related vars are None

        # --- 1. Reconstruction Loss for X (MIWAE style) ---
        num_observed_x = x_mask_batch.sum().clamp(min=1)
        if self.reconstruction_loss_type_x == 'huber':
            elementwise_loss_x = F.huber_loss(recon_x, x_batch, reduction='none', delta=self.huber_delta)
        else: # Default 'mse'
            elementwise_loss_x = F.mse_loss(recon_x, x_batch, reduction='none')
        recon_loss_x = (elementwise_loss_x * x_mask_batch).sum() / num_observed_x

        # --- 2. KL Divergence ---
        kld_loss = -0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp(), dim=1).mean()
        if torch.isnan(kld_loss) or torch.isinf(kld_loss):
            self.logger.error(f"KLD loss is {kld_loss.item()}. Inputs: mu_mean={mu_latent.mean():.2f}, logvar_mean={logvar_latent.mean():.2f}. Setting KLD to 0.")
            kld_loss = torch.tensor(0.0, device=self.device)

        # --- 3. Supervised Prediction Loss for Y ---
        loss_y = torch.tensor(0.0, device=self.device)
        if self.is_semi_supervised and y_batch is not None and y_mask_batch is not None:
             # Ensure all required tensors for Y loss are present
             if y_pred_mu is not None and y_pred_logvar is not None:
                 if y_mask_batch.sum() > 0: # Calculate only if observed Y exist
                     loss_y = self._calculate_y_loss(y_batch, y_pred_mu, y_pred_logvar, y_mask_batch)
             else: # Should not happen if model is SemiSupMIWAE, but safeguard
                 self.logger.error("Semi-supervised mode, but Y predictions (mu/logvar) are missing from model output.")
        # else: loss_y remains 0 for unsupervised or if Y data/predictions are missing

        # --- Total Loss (Negative ELBO + Supervised Loss) ---
        # Loss = ReconLoss_X + beta * KLD + alpha * Loss_Y
        total_loss = recon_loss_x + current_beta_kld * kld_loss + self.alpha_price_loss * loss_y

        if torch.isnan(total_loss):
             self.logger.error(f"Total loss is NaN! Components -> ReconX: {recon_loss_x.item()}, KLD: {kld_loss.item()}, LossY: {loss_y.item()}. Skipping update.")
             # Potentially return sentinel or raise error depending on desired behavior
             # For now, let it propagate and potentially get caught in train loop

        return total_loss, recon_loss_x, kld_loss, loss_y # Return components for logging

    def train(self,
              train_dataset: TensorDataset,
              val_dataset: Optional[TensorDataset] = None,
              epochs: int = 50,
              batch_size: int = 64,
              kld_anneal_epochs: int = 0,
              early_stopping_patience: Optional[int] = None,
              print_every_n_epochs: int = 1,
              scheduler_patience: int = 5,
              scheduler_factor: float = 0.2
             ) -> Dict[str, List[float]]:
        """Trains the VAE model."""
        self._log(f"Starting training: Epochs={epochs}, BatchSize={batch_size}, "
                  f"KLD_weight={self.kld_weight}, KLD_AnnealEpochs={kld_anneal_epochs}, "
                  f"Alpha(Y_Loss)={self.alpha_price_loss if self.is_semi_supervised else 'N/A'}.",
                  level="info")

        # Setup DataLoaders
        use_pin_memory = self.device.type == 'cuda'
        num_workers = min(4, os.cpu_count() or 1) # Sensible default
        should_drop_last = (len(train_dataset) % batch_size == 1) and getattr(self.model, 'use_batch_norm', False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=should_drop_last, num_workers=num_workers, pin_memory=use_pin_memory)
        self._log(f"Train loader: {len(train_dataset)} samples, {len(train_loader)} batches. Drop last: {should_drop_last}")

        val_loader = None; best_val_loss = float('inf'); epochs_no_improve = 0
        if val_dataset:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=use_pin_memory)
            self._log(f"Validation loader: {len(val_dataset)} samples, {len(val_loader)} batches.")
        else:
            self._log("No validation dataset. Early stopping/best model saving disabled.")
            early_stopping_patience = None # Disable if no val data

        # Setup Scheduler
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=scheduler_factor,
                                      patience=scheduler_patience)

        # --- Training Loop ---
        for epoch in range(epochs):
            self.model.train() # Set model to training mode
            epoch_losses = defaultdict(float) # Accumulate losses for the epoch average
            num_batches_processed = 0
            start_time_epoch = time.time()

            # KLD annealing factor (beta) for this epoch
            current_beta_kld = self.kld_weight * min(1.0, (epoch + 1) / kld_anneal_epochs) if kld_anneal_epochs > 0 else self.kld_weight

            for batch_idx, batch_data in enumerate(train_loader):
                # --- Batch Processing ---
                try:
                    # Unpack data and move to device
                    expected_tensors = 4 if self.is_semi_supervised else 2
                    if not isinstance(batch_data, (list, tuple)) or len(batch_data) != expected_tensors:
                        raise TypeError(f"DataLoader returned {len(batch_data)} tensors, expected {expected_tensors}")

                    if self.is_semi_supervised:
                        x_b, x_m_b, y_b, y_m_b = [t.to(self.device) for t in batch_data]
                    else:
                        x_b, x_m_b = [t.to(self.device) for t in batch_data]
                        y_b, y_m_b = None, None

                    # Skip batches potentially causing BN issues (safeguard)
                    if x_b.shape[0] <= 1 and getattr(self.model, 'use_batch_norm', False): continue

                    # Forward pass, loss calculation, backward pass, optimizer step
                    self.optimizer.zero_grad()
                    model_outputs = self.model(x_b)
                    loss, recon_lx, kld_l, loss_ly = self._calculate_loss(
                        x_b, x_m_b, y_b, y_m_b, model_outputs, current_beta_kld
                    )

                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"NaN/Inf loss in Epoch {epoch+1}/Batch {batch_idx}. Skipping update. "
                                            f"Loss: {loss.item()}, ReconX: {recon_lx.item()}, KLD: {kld_l.item()}, LossY: {loss_ly.item()}")
                        continue # Skip optimizer step if loss is invalid

                    loss.backward()
                    self.optimizer.step()

                    # Accumulate losses for epoch average
                    epoch_losses['total_loss'] += loss.item()
                    epoch_losses['recon_loss_x'] += recon_lx.item()
                    epoch_losses['kld_loss'] += kld_l.item()
                    epoch_losses['loss_y'] += loss_ly.item() # loss_ly is 0 if not applicable
                    num_batches_processed += 1

                except Exception as batch_err:
                    self.logger.error(f"Error in training batch {epoch+1}/{batch_idx}: {batch_err}", exc_info=False)
                    continue # Skip to next batch on error
            # --- End of Batch Loop ---

            if num_batches_processed == 0:
                self.logger.warning(f"Epoch {epoch+1}: No batches processed successfully. Skipping epoch summary.")
                continue

            # Calculate average losses for the epoch
            avg_train_total_loss = epoch_losses['total_loss'] / num_batches_processed
            avg_train_recon_loss_x = epoch_losses['recon_loss_x'] / num_batches_processed
            avg_train_kld_loss = epoch_losses['kld_loss'] / num_batches_processed
            avg_train_loss_y = epoch_losses['loss_y'] / num_batches_processed

            # Record training history
            self.history['train_loss'].append(avg_train_total_loss)
            self.history['train_recon_loss_x'].append(avg_train_recon_loss_x)
            self.history['train_kld_loss'].append(avg_train_kld_loss)
            if self.is_semi_supervised: self.history['train_loss_y'].append(avg_train_loss_y)

            epoch_duration = time.time() - start_time_epoch
            # Construct train log message
            log_msg_parts = [
                f"Epoch [{epoch+1}/{epochs}] Train TotalLoss: {avg_train_total_loss:.4f}",
                f"(ReconX: {avg_train_recon_loss_x:.4f}",
                f"KLD: {avg_train_kld_loss:.4f}*{current_beta_kld:.2f}"
            ]
            if self.is_semi_supervised: log_msg_parts.append(f"LossY: {avg_train_loss_y:.4f}*{self.alpha_price_loss:.2f}")
            log_msg_parts.append(f"), Time: {epoch_duration:.1f}s")
            train_log_message = ", ".join(log_msg_parts[:-1]) + log_msg_parts[-1] # Combine nicely

            # --- Validation Step ---
            current_val_loss_for_scheduler = avg_train_total_loss # Use train loss if no validation
            if val_loader:
                self.model.eval() # Set to evaluation mode
                val_losses = defaultdict(float)
                val_batches_processed = 0
                with torch.no_grad():
                    for val_batch_data in val_loader:
                        try:
                            expected_tensors = 4 if self.is_semi_supervised else 2
                            if not isinstance(val_batch_data, (list, tuple)) or len(val_batch_data) != expected_tensors: continue # Skip malformed batch

                            if self.is_semi_supervised:
                                x_v, x_m_v, y_v, y_m_v = [t.to(self.device) for t in val_batch_data]
                            else:
                                x_v, x_m_v = [t.to(self.device) for t in val_batch_data]
                                y_v, y_m_v = None, None
                            if x_v.shape[0] <= 1 and getattr(self.model, 'use_batch_norm', False): continue

                            val_model_outputs = self.model(x_v)
                            v_loss, v_recon_x, v_kld, v_loss_y = self._calculate_loss(
                                x_v, x_m_v, y_v, y_m_v, val_model_outputs, self.kld_weight # Use full beta for val loss
                            )
                            if not (torch.isnan(v_loss) or torch.isinf(v_loss)):
                                val_losses['total_loss'] += v_loss.item()
                                val_losses['recon_loss_x'] += v_recon_x.item()
                                val_losses['kld_loss'] += v_kld.item()
                                val_losses['loss_y'] += v_loss_y.item()
                                val_batches_processed += 1
                        except Exception as val_err:
                            self.logger.warning(f"Error in validation batch: {val_err}", exc_info=False)
                            continue

                if val_batches_processed > 0:
                    avg_val_total_loss = val_losses['total_loss'] / val_batches_processed
                    avg_val_recon_loss_x = val_losses['recon_loss_x'] / val_batches_processed
                    avg_val_kld_loss = val_losses['kld_loss'] / val_batches_processed
                    avg_val_loss_y = val_losses['loss_y'] / val_batches_processed
                    current_val_loss_for_scheduler = avg_val_total_loss # Use this for scheduler/early stopping

                    # Record validation history
                    self.history['val_loss'].append(avg_val_total_loss)
                    self.history['val_recon_loss_x'].append(avg_val_recon_loss_x)
                    self.history['val_kld_loss'].append(avg_val_kld_loss)
                    if self.is_semi_supervised: self.history['val_loss_y'].append(avg_val_loss_y)

                    # Append validation info to log message
                    val_log_parts = [
                         f"| Val TotalLoss: {avg_val_total_loss:.4f}",
                         f"(ReconX: {avg_val_recon_loss_x:.4f}",
                         f"KLD: {avg_val_kld_loss:.4f}"
                    ]
                    if self.is_semi_supervised: val_log_parts.append(f"LossY: {avg_val_loss_y:.4f}")
                    val_log_parts.append(")")
                    train_log_message += " " + ", ".join(val_log_parts[:-1]) + val_log_parts[-1]
                else: # No valid validation batches processed
                    self.history['val_loss'].append(float('nan')) # Log NaN if val failed
                    self.history['val_recon_loss_x'].append(float('nan'))
                    self.history['val_kld_loss'].append(float('nan'))
                    if self.is_semi_supervised: self.history['val_loss_y'].append(float('nan'))
                    train_log_message += " | Val Loss: N/A"
                    current_val_loss_for_scheduler = float('nan') # Cannot use for scheduler/early stop

            # --- Logging, Scheduler, Early Stopping ---
            if (epoch + 1) % print_every_n_epochs == 0 or epoch == epochs - 1 or \
               (early_stopping_patience and epochs_no_improve > 0): # Log if patience counting
                self._log(train_log_message, level="info")

            if not np.isnan(current_val_loss_for_scheduler): # Only step if val loss is valid
                scheduler.step(current_val_loss_for_scheduler)
                if early_stopping_patience is not None:
                    if current_val_loss_for_scheduler < best_val_loss:
                        best_val_loss = current_val_loss_for_scheduler; epochs_no_improve = 0
                        self.save_model(filename="best_model.pth") # Save best model state
                        self._log(f"Epoch {epoch+1}: New best val_loss: {best_val_loss:.4f}. Model saved.", level="debug")
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= early_stopping_patience:
                            self._log(f"Early stopping triggered at epoch {epoch+1}. Best val_loss: {best_val_loss:.4f}")
                            break # Exit training loop
        # --- End of Epoch Loop ---

        self._log("Training loop finished.", level="info")
        # Load best model if early stopping was active and saved a model
        best_model_path = os.path.join(self.results_dir, "best_model.pth")
        if early_stopping_patience is not None and os.path.exists(best_model_path):
            self._log(f"Loading best model state from '{best_model_path}' (Val Loss: {best_val_loss:.4f}).")
            try:
                 self.load_model(filepath=best_model_path, device=self.device) # Load the best weights back into self.model
            except Exception as e_load:
                 self._log(f"Failed to load best model: {e_load}. Continuing with final epoch model.", level="error")
                 self.save_model(filename="final_model_epoch_end.pth") # Save final weights instead
        else: # No early stopping or no best model saved
            self.save_model(filename="final_model_epoch_end.pth") # Save the model from the last epoch

        return dict(self.history) # Return training history

    def predict_price_and_uncertainty(self, X_filled_tensor: torch.Tensor, batch_size: int = 1024) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Predicts target Y mean and log-variance using a trained SemiSupMIWAE model."""
        if not self.is_semi_supervised or not isinstance(self.model, SemiSupMIWAE):
            self.logger.error("Price prediction requires a trained SemiSupMIWAE model.")
            return None, None

        self._log(f"Predicting target (Y) mean and log-variance for {X_filled_tensor.shape[0]} samples...", level="info")
        self.model.eval() # Set model to evaluation mode

        # Create a dataset/loader for efficient batch processing
        pred_dataset = TensorDataset(X_filled_tensor)
        pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)

        all_y_pred_mu, all_y_pred_logvar = [], []
        with torch.no_grad():
            for (x_batch,) in pred_loader: # DataLoader wraps tensor in a tuple
                x_batch = x_batch.to(self.device)
                try:
                    # Use the model's dedicated prediction method if available
                    if hasattr(self.model, 'predict_price') and callable(getattr(self.model, 'predict_price')):
                        y_mu_batch, y_logvar_batch = self.model.predict_price(x_batch)
                    else: # Fallback (should not be needed for SemiSupMIWAE)
                         mu_latent, _ = self.model.encode(x_batch)
                         y_mu_batch = self.model.price_mean_head(mu_latent)
                         y_logvar_batch = self.model.price_logvar_head(mu_latent)

                    all_y_pred_mu.append(y_mu_batch.cpu().numpy())
                    all_y_pred_logvar.append(y_logvar_batch.cpu().numpy())
                except Exception as e_pred:
                    self.logger.error(f"Error during batch Y prediction: {e_pred}", exc_info=True)
                    return None, None # Abort prediction

        if not all_y_pred_mu: # Check if any predictions were made
             self.logger.warning("Y prediction process generated no output.")
             return None, None

        # Concatenate results from all batches
        try:
            y_mu_full = np.concatenate(all_y_pred_mu, axis=0)
            y_logvar_full = np.concatenate(all_y_pred_logvar, axis=0)
        except ValueError as e_concat:
            self._log(f"Error concatenating Y prediction results: {e_concat}. Check batch output shapes.", level="error")
            return None, None

        self._log(f"Y prediction complete. Output shapes: y_mu={y_mu_full.shape}, y_logvar={y_logvar_full.shape}", level="debug")
        return y_mu_full, y_logvar_full

    def impute(self, X_missing_np: np.ndarray, n_imputations: int = 1) -> Union[np.ndarray, List[np.ndarray]]:
        """Imputes missing values (NaN) in X_missing_np using the trained VAE model."""
        if self.model is None:
            raise RuntimeError("Model is not loaded or trained. Cannot impute.")
        self._log(f"Starting imputation for X data ({X_missing_np.shape[0]} samples, {X_missing_np.shape[1]} features), n_imputations={n_imputations}...", level="info")
        self.model.eval() # Set to evaluation mode

        nan_mask = np.isnan(X_missing_np)
        if not np.any(nan_mask):
            self._log("No missing values (NaNs) found in input X data. Returning copy/copies of original.", level="info")
            return [X_missing_np.copy() for _ in range(n_imputations)] if n_imputations > 1 else X_missing_np.copy()

        # Fill NaNs with 0.0 for model input
        X_filled_for_input_np = np.nan_to_num(X_missing_np, nan=0.0)
        # Ensure input dimension matches model's expected input_dim for X
        if X_filled_for_input_np.shape[1] != self.model.input_dim:
             self._log(f"Input data feature dimension for imputation ({X_filled_for_input_np.shape[1]}) "
                       f"does not match model input dimension ({self.model.input_dim}). Cannot impute.", level="error")
             raise ValueError("Input dimension mismatch during imputation.")

        X_filled_for_input_tensor = torch.from_numpy(X_filled_for_input_np).float().to(self.device)

        imputed_datasets = []
        with torch.no_grad():
            for i in range(n_imputations):
                # MIWAE imputation: Encode -> Reparameterize -> Decode X
                mu, logvar = self.model.encode(X_filled_for_input_tensor)
                z = self.model.reparameterize(mu, logvar)
                reconstructed_x_tensor = self.model.decode_x(z)
                reconstructed_x_np = reconstructed_x_tensor.cpu().numpy()

                # Create imputed dataset: start with original, fill NaNs with reconstruction
                current_imputed_dataset_np = X_missing_np.copy()
                current_imputed_dataset_np[nan_mask] = reconstructed_x_np[nan_mask]
                imputed_datasets.append(current_imputed_dataset_np)

                if n_imputations > 1 and (i + 1) % max(1, n_imputations // 5) == 0: # Log progress
                    self._log(f"Completed imputation {i+1}/{n_imputations}", level="debug")

        self._log(f"Imputation complete. Returning {len(imputed_datasets)} imputed dataset(s).", level="info")
        return imputed_datasets[0] if n_imputations == 1 else imputed_datasets

    def save_model(self, filename: str = "vae_model_final.pth"):
        """Saves the model state, configuration, optimizer state, and history."""
        if self.model is None: self._log("No model instance to save.", level="warning"); return
        filepath = os.path.join(self.results_dir, filename)
        self._log(f"Saving model and trainer state to {filepath}...", level="info")
        try:
            # --- Extract Model Configuration ---
            # Basic architecture
            encoder_layers = [l.out_features for l in getattr(self.model, 'encoder_net', []) if isinstance(l, nn.Linear)]
            decoder_layers_all = [l for l in getattr(self.model, 'decoder_net_x', []) if isinstance(l, nn.Linear)]
            decoder_layers = [l.out_features for l in decoder_layers_all[:-1]] if len(decoder_layers_all)>1 else [] # Exclude final output layer

            model_config = {
                'model_class': self.model.__class__.__name__,
                'input_dim': getattr(self.model, 'input_dim', None), # X dim
                'latent_dim': getattr(self.model, 'latent_dim', None),
                'encoder_layer_sizes': encoder_layers,
                'decoder_layer_sizes': decoder_layers,
                'activation_fn_str': getattr(getattr(self.model, 'activation_fn', nn.ReLU()), '__class__', nn.Module).__name__.lower(),
                'use_batch_norm': getattr(self.model, 'use_batch_norm', True),
                'prior_type': getattr(self.model, 'prior_type', 'standard_gaussian'),
                'n_prior_components': getattr(self.model, 'n_prior_components', 1),
                'learn_prior': getattr(self.model, 'learn_prior', False),
            }
            # Add SemiSupMIWAE specific parts if applicable
            if isinstance(self.model, SemiSupMIWAE):
                model_config['y_dim'] = getattr(self.model, 'y_dim', 1)
                price_head_layers_all = [l for l in getattr(self.model, 'price_mean_head', []) if isinstance(l, nn.Linear)]
                model_config['price_head_layer_sizes'] = [l.out_features for l in price_head_layers_all[:-1]] if len(price_head_layers_all)>1 else []
                model_config['alpha_price_loss'] = getattr(self.model, 'alpha_price_loss', 1.0) # Model's alpha

            final_model_config = {k: v for k, v in model_config.items() if v is not None} # Remove None values

            # --- Extract Trainer Configuration ---
            trainer_config = {
                'learning_rate': self.optimizer.defaults.get('lr'),
                'kld_weight': self.kld_weight, # Base KLD weight used for training/val loss reporting
                'alpha_price_loss_trainer': self.alpha_price_loss, # Trainer's effective alpha
                'optimizer_str': self.optimizer.__class__.__name__.lower(),
                'reconstruction_loss_type_x': self.reconstruction_loss_type_x,
                'price_loss_type_y': self.price_loss_type_y,
                'huber_delta': self.huber_delta,
            }
            final_trainer_config = {k: v for k,v in trainer_config.items() if v is not None}

            # --- Prepare Save Content ---
            save_content = {
                'model_state_dict': self.model.state_dict(),
                'config': final_model_config, # Model architecture config
                'optimizer_state_dict': self.optimizer.state_dict(),
                'history': dict(self.history), # Convert defaultdict for saving
                'trainer_config': final_trainer_config # Trainer setup config
            }
            torch.save(save_content, filepath)
            self._log(f"Successfully saved model and trainer state to {filepath}", level="info")

        except Exception as e:
            self._log(f"Error saving model state to {filepath}: {e}", level="error", exc_info=True)

    @classmethod
    def load_model(cls, filepath: str, device: Optional[torch.device] = None, verbose: bool = True) -> 'VAETrainer':
        """Loads a saved VAE model and trainer state."""
        logger_loader = get_logger(cls.__name__ + "_Loader", verbose)
        effective_device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(filepath):
            logger_loader.error(f"Model checkpoint file not found: {filepath}")
            raise FileNotFoundError(f"VAE model file not found: {filepath}")

        logger_loader.info(f"Loading VAE model and trainer state from {filepath} to device {effective_device}...")
        try:
            checkpoint = torch.load(filepath, map_location=effective_device)

            # --- Reconstruct Model ---
            config = checkpoint.get('config', {})
            if not config: raise ValueError("Model configuration ('config') missing in checkpoint.")
            model_class_name = config.get('model_class', 'VariationalAutoencoderBase')

            # Activation function
            activation_map = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(), 'elu': nn.ELU(), 'selu': nn.SELU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}
            activation_fn = activation_map.get(config.get('activation_fn_str','relu'), nn.ReLU())

            # Basic VAE arguments from config
            model_args = {
                k: config.get(k) for k in ['input_dim', 'latent_dim', 'encoder_layer_sizes', 'decoder_layer_sizes']
            }
            if None in model_args.values(): raise ValueError(f"Missing core VAE dimensions in config: {model_args}")
            model_args.update({
                'activation_fn': activation_fn,
                'use_batch_norm': config.get('use_batch_norm', True),
                'prior_type': config.get('prior_type', 'standard_gaussian'),
                'n_prior_components': config.get('n_prior_components', 1),
                'learn_prior': config.get('learn_prior', False),
                'device': effective_device, # Pass device for model instantiation
            })

            ModelClass: Union[type(VariationalAutoencoderBase), type(SemiSupMIWAE)]
            if model_class_name == 'SemiSupMIWAE':
                ModelClass = SemiSupMIWAE
                model_args.update({ # Add SemiSupMIWAE specific args
                    'y_dim': config.get('y_dim', 1),
                    'price_head_layer_sizes': config.get('price_head_layer_sizes', []),
                    'alpha_price_loss': config.get('alpha_price_loss', 1.0) # Use model's saved alpha
                })
            elif model_class_name == 'VariationalAutoencoderBase':
                ModelClass = VariationalAutoencoderBase
            else:
                raise ValueError(f"Unknown model_class '{model_class_name}' in checkpoint.")

            model_instance = ModelClass(**model_args)
            model_instance.load_state_dict(checkpoint['model_state_dict'])
            model_instance.to(effective_device) # Ensure model is on the correct device
            logger_loader.info(f"{ModelClass.__name__} model created and state loaded successfully.")

            # --- Reconstruct Trainer ---
            trainer_config = checkpoint.get('trainer_config', {}) # Get trainer config saved previously
            trainer_instance = cls(
                model=model_instance,
                learning_rate=trainer_config.get('learning_rate', 1e-3),
                kld_weight=trainer_config.get('kld_weight', 1.0),
                # Use alpha from trainer_config if saved, otherwise None (trainer init will handle)
                alpha_price_loss=trainer_config.get('alpha_price_loss_trainer'),
                optimizer_str=trainer_config.get('optimizer_str', 'adam'),
                reconstruction_loss_type_x=trainer_config.get('reconstruction_loss_type_x', 'mse'),
                price_loss_type_y=trainer_config.get('price_loss_type_y', 'gaussian_nll'),
                huber_delta=trainer_config.get('huber_delta', 1.0),
                device=effective_device,
                results_dir=os.path.dirname(filepath) or ".", # Use directory of loaded file
                verbose=verbose
            )

            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                try:
                    trainer_instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger_loader.debug("Optimizer state loaded successfully.")
                except Exception as e_optim:
                    logger_loader.warning(f"Could not load optimizer state_dict: {e_optim}. Optimizer remains re-initialized.")
            else:
                logger_loader.warning("Optimizer state_dict not found in checkpoint. Optimizer remains re-initialized.")

            # Restore history
            trainer_instance.history = defaultdict(list, checkpoint.get('history', {}))
            logger_loader.info(f"VAETrainer instance created and state loaded for model from {filepath}.")
            return trainer_instance

        except Exception as e:
            logger_loader.error(f"Critical error loading VAE model and trainer state from {filepath}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load VAE model and trainer from {filepath}") from e

    def get_latent_representation(self, X_np: np.ndarray, batch_size: int = 256) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Encodes input data X_np into the latent space (mu, logvar). Handles NaN imputation."""
        if self.model is None:
             self._log("Model not available for latent representation.", level="error"); return None
        self.model.eval()

        # Impute NaNs for encoding purposes
        if np.any(np.isnan(X_np)):
            self._log("Input data for latent encoding contains NaNs. Filling with 0.0.", level="debug")
            X_filled_np = np.nan_to_num(X_np, nan=0.0).astype(np.float32)
        else:
            X_filled_np = X_np.astype(np.float32)

        # Check dimensions
        if X_filled_np.shape[1] != self.model.input_dim:
            self._log(f"Input data dim ({X_filled_np.shape[1]}) != model input dim ({self.model.input_dim}).", level="error")
            return None

        # Batch processing for potentially large inputs
        dataset = TensorDataset(torch.from_numpy(X_filled_np))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_mu, all_logvar = [], []

        with torch.no_grad():
            for (x_batch_tensor,) in loader:
                try:
                    mu_batch, logvar_batch = self.model.encode(x_batch_tensor.to(self.device))
                    all_mu.append(mu_batch.cpu().numpy())
                    all_logvar.append(logvar_batch.cpu().numpy())
                except Exception as e_encode:
                    self._log(f"Error during batch encoding for latent representation: {e_encode}", level="error", exc_info=True)
                    return None

        if not all_mu: return None # No batches processed

        # Concatenate results
        try:
            mu_full = np.concatenate(all_mu, axis=0)
            logvar_full = np.concatenate(all_logvar, axis=0)
        except ValueError as e_concat:
            self._log(f"Error concatenating latent representations: {e_concat}.", level="error"); return None

        self._log(f"Encoded data to latent space. Mu shape: {mu_full.shape}, LogVar shape: {logvar_full.shape}", level="debug")
        return mu_full, logvar_full

