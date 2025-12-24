# -*- coding: utf-8 -*-
"""
Variational Autoencoder base class with configurable prior (Gaussian, Student-t, Mixture).
"""
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal, Categorical, Independent, Distribution

from probml.core.utils import get_logger
from probml.models.student_t import StudentTDistribution
from probml.models.distributions import MixtureDistribution


# --- Base VAE Architecture ---
class VariationalAutoencoderBase(nn.Module):
    def __init__(self,
                 input_dim: int,
                 encoder_layer_sizes: List[int],
                 latent_dim: int,
                 decoder_layer_sizes: List[int],
                 activation_fn: nn.Module = nn.ReLU(),
                 use_batch_norm: bool = True,
                 prior_type: str = 'standard_gaussian',
                 n_prior_components: int = 1,
                 prior_params: Optional[Dict[str, Any]] = None,
                 learn_prior: bool = False,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm
        self.prior_type = prior_type
        self.n_prior_components = max(1, n_prior_components) # Ensure at least 1 component
        self.learn_prior = learn_prior
        self.device_ = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger(f"{self.__class__.__name__}(In:{input_dim},Lat:{latent_dim})", verbose=True)

        # Encoder
        encoder_modules = []
        current_dim_enc = input_dim
        for i, layer_size in enumerate(encoder_layer_sizes):
            encoder_modules.append(nn.Linear(current_dim_enc, layer_size))
            if self.use_batch_norm:
                encoder_modules.append(nn.BatchNorm1d(layer_size))
            encoder_modules.append(activation_fn)
            current_dim_enc = layer_size
        self.encoder_net = nn.Sequential(*encoder_modules)
        self.fc_mu = nn.Linear(current_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(current_dim_enc, latent_dim)

        # Decoder (Corrected logic from original prompt for clarity)
        decoder_main_modules = []
        current_dim_dec = latent_dim
        for i, layer_size in enumerate(decoder_layer_sizes):
            decoder_main_modules.append(nn.Linear(current_dim_dec, layer_size))
            if self.use_batch_norm:
                decoder_main_modules.append(nn.BatchNorm1d(layer_size))
            decoder_main_modules.append(activation_fn)
            current_dim_dec = layer_size

        # Final output layer for X reconstruction
        self.decoder_output_x = nn.Linear(current_dim_dec, input_dim)
        decoder_main_modules.append(self.decoder_output_x)
        self.decoder_net_x = nn.Sequential(*decoder_main_modules)

        self._setup_prior(prior_params if prior_params else {})
        self.to(self.device_)
        self.logger.debug(f"VAEBase: InputDim={input_dim}, Enc={encoder_layer_sizes}, Latent={latent_dim}, Dec={decoder_layer_sizes}, Prior={prior_type}(K={n_prior_components}) on {self.device_}")


    def _init_param(self, data: Optional[Any], default_shape: Tuple[int, ...], default_val: Any) -> Union[torch.Tensor, nn.Parameter]:
        param_wrapper = nn.Parameter if self.learn_prior else lambda x: x # Apply nn.Parameter if prior is learnable
        val_to_wrap = None
        data_tensor: Optional[torch.Tensor] = None

        if data is not None:
            if not isinstance(data, torch.Tensor):
                try:
                    data_tensor = torch.tensor(data, dtype=torch.float32) # Device set later
                except Exception as e:
                    self.logger.warning(f"Could not convert prior param data to tensor: {e}. Defaulting.")
                    data_tensor = None
            else:
                data_tensor = data.clone().detach() # Use clone of input tensor

            if data_tensor is not None:
                if not torch.isfinite(data_tensor).all():
                    self.logger.warning(f"Non-finite values in provided prior param data. Defaulting. Data: {data_tensor}")
                    data_tensor = None
                elif data_tensor.shape == default_shape:
                    val_to_wrap = data_tensor
                elif data_tensor.numel() == 1 and default_shape != torch.Size([]): # Scalar to expand
                    val_to_wrap = data_tensor.expand(default_shape)
                else:
                    self.logger.warning(f"Prior param shape mismatch (Provided: {data_tensor.shape}, Expected: {default_shape}). Defaulting.")
                    data_tensor = None # Signal to use default

        if val_to_wrap is None: # Use default_val
            if isinstance(default_val, torch.Tensor):
                if default_val.shape == default_shape:
                    val_to_wrap = default_val.clone().detach()
                elif default_val.numel() == 1 and default_shape != torch.Size([]):
                    val_to_wrap = default_val.expand(default_shape)
                else: # Should not happen if defaults are well-defined
                    self.logger.error(f"Default tensor shape {default_val.shape} vs expected {default_shape}. Using zeros.")
                    val_to_wrap = torch.zeros(default_shape, dtype=torch.float32)
            else: # default_val is a scalar (e.g. float, int)
                val_to_wrap = torch.full(default_shape, float(default_val), dtype=torch.float32)

        return param_wrapper(val_to_wrap.to(self.device_))


    def _setup_prior(self, prior_params_dict: Dict[str, Any]):
        # Default parameters are defined on the target device directly
        default_mean = torch.zeros(self.n_prior_components, self.latent_dim, device=self.device_)
        default_cov_chol = torch.eye(self.latent_dim, device=self.device_).unsqueeze(0).expand(self.n_prior_components, self.latent_dim, self.latent_dim)
        default_weights_logits = torch.zeros(self.n_prior_components, device=self.device_)
        default_df_student_t = torch.full((self.n_prior_components, self.latent_dim), 4.0, device=self.device_) # Default DoF for Student-t

        self.prior_weights_logits = self._init_param(prior_params_dict.get('weights_logits'), (self.n_prior_components,), default_weights_logits)
        self.prior_means = self._init_param(prior_params_dict.get('means'), (self.n_prior_components, self.latent_dim), default_mean)
        self.prior_cov_cholesky = self._init_param(prior_params_dict.get('cov_cholesky'), (self.n_prior_components, self.latent_dim, self.latent_dim), default_cov_chol)

        self.prior_student_t_df = None # Initialize
        if 'student_t' in self.prior_type.lower():
            self.prior_student_t_df = self._init_param(prior_params_dict.get('df'), (self.n_prior_components, self.latent_dim), default_df_student_t)
        self.logger.debug(f"Prior setup: Type='{self.prior_type}', K={self.n_prior_components}, Learnable={self.learn_prior}")

    def get_prior_distribution(self) -> Distribution:
        # Ensure parameters are on the correct device, though _init_param should handle this.
        # Accessing them directly here, e.g., self.prior_means.to(self.device_) is redundant if _init_param is correct.
        try:
            if self.n_prior_components == 1 and 'mixture' not in self.prior_type.lower(): # Single component prior
                loc = self.prior_means.squeeze(0) # Shape: (latent_dim)
                chol = self.prior_cov_cholesky.squeeze(0) # Shape: (latent_dim, latent_dim)

                if 'student_t' in self.prior_type.lower():
                    if self.prior_student_t_df is None: raise ValueError("Student-T prior_student_t_df is None.")
                    df = self.prior_student_t_df.squeeze(0).clamp(min=1e-6) # Shape: (latent_dim)

                    # For Independent StudentT, scale is per dimension
                    cov_matrix_diag = torch.sum(chol * chol, dim=-1) # Diagonal elements of covariance if chol is diagonal
                                                                    # If chol is full, (chol @ chol.T).diag()
                    if chol.shape == (self.latent_dim, self.latent_dim) and torch.allclose(chol, torch.diag(torch.diag(chol))): # Is diagonal
                        scales = torch.diag(chol).clamp(min=1e-6)
                    else: # Full Cholesky, get diagonal variances
                         scales = torch.sqrt((chol @ chol.transpose(-1, -2)).diag()).clamp(min=1e-6)

                    if loc.ndim == 0: loc = loc.unsqueeze(0) # Ensure 1D for component dist
                    if scales.ndim == 0: scales = scales.unsqueeze(0)
                    if df.ndim == 0: df = df.unsqueeze(0)

                    base_dist = StudentTDistribution(df=df, loc=loc, scale=scales) # Batch shape (latent_dim), event_shape ()
                    return Independent(base_dist, 1) # Makes event_shape (latent_dim)
                else: # Single Multivariate Gaussian
                    return MultivariateNormal(loc=loc, scale_tril=chol)
            else: # Mixture prior
                mix = Categorical(logits=self.prior_weights_logits) # Mixture weights, shape (K)

                if 'student_t' in self.prior_type.lower():
                    if self.prior_student_t_df is None: raise ValueError("Student-T mixture prior_student_t_df is None.")
                    df_mix = self.prior_student_t_df.clamp(min=1e-6) # Shape: (K, latent_dim)
                    loc_mix = self.prior_means # Shape: (K, latent_dim)

                    # For Independent StudentT components in a mixture
                    # Assuming self.prior_cov_cholesky are Cholesky factors of *diagonal* covariances for each component
                    chol_mix = self.prior_cov_cholesky # Shape: (K, latent_dim, latent_dim)
                    if torch.allclose(chol_mix, torch.diag_embed(torch.diagonal(chol_mix, dim1=-2, dim2=-1))): # Check if all components have diagonal chols
                        scales_mix = torch.diagonal(chol_mix, dim1=-2, dim2=-1).clamp(min=1e-6) # Shape: (K, latent_dim)
                    else: # If full cholesky per component
                         scales_mix = torch.sqrt(torch.diagonal(chol_mix @ chol_mix.transpose(-1,-2), dim1=-2, dim2=-1)).clamp(min=1e-6)

                    base_dist_mix = StudentTDistribution(df=df_mix, loc=loc_mix, scale=scales_mix) # Batch_shape (K, latent_dim), event_shape ()
                    comp = Independent(base_dist_mix, 1) # Makes component event_shape (latent_dim), batch_shape (K)
                else: # Gaussian Mixture
                    comp = MultivariateNormal(loc=self.prior_means, scale_tril=self.prior_cov_cholesky) # Batch_shape (K), event_shape (latent_dim)

                return MixtureDistribution(mix, comp)
        except Exception as e:
            self.logger.error(f"Error creating prior distribution (type: {self.prior_type}, K: {self.n_prior_components}): {e}. Defaulting to Standard Normal.", exc_info=True)
            return Independent(Normal(torch.zeros(self.latent_dim, device=self.device_),
                                      torch.ones(self.latent_dim, device=self.device_)), 1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_x(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent variable z to reconstruct X (input features)."""
        return self.decoder_net_x(z)

