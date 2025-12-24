# -*- coding: utf-8 -*-
from probml.core.utils import get_logger
from probml.models.vae_base import VariationalAutoencoderBase


# --- Semi-Supervised MIWAE Model (Includes Price Heads) ---
class SemiSupMIWAE(VariationalAutoencoderBase):
    def __init__(self,
                 input_dim: int, # Dimension of X features
                 y_dim: int,     # Dimension of Y target (e.g., price, typically 1)
                 encoder_layer_sizes: List[int],
                 latent_dim: int,
                 decoder_layer_sizes: List[int], # For X reconstruction
                 price_head_layer_sizes: List[int], # For Y prediction
                 activation_fn: nn.Module = nn.ReLU(),
                 use_batch_norm: bool = True,
                 prior_type: str = 'standard_gaussian',
                 n_prior_components: int = 1,
                 prior_params: Optional[Dict[str, Any]] = None,
                 learn_prior: bool = False,
                 alpha_price_loss: float = 1.0, # Weight for the price prediction loss
                 device: Optional[torch.device] = None):
        super().__init__(input_dim=input_dim, encoder_layer_sizes=encoder_layer_sizes,
                         latent_dim=latent_dim, decoder_layer_sizes=decoder_layer_sizes,
                         activation_fn=activation_fn, use_batch_norm=use_batch_norm,
                         prior_type=prior_type, n_prior_components=n_prior_components,
                         prior_params=prior_params, learn_prior=learn_prior, device=device)
        self.y_dim = y_dim
        self.alpha_price_loss = alpha_price_loss
        # Update logger name for clarity
        self.logger = get_logger(f"{self.__class__.__name__}(X:{input_dim},Y:{y_dim},Lat:{latent_dim})", verbose=True)


        # Price Prediction Heads (operate on latent mu)
        price_mean_modules = []
        price_logvar_modules = [] # Separate head for uncertainty (logvariance)

        current_dim_price_head = latent_dim # Input to price head is the latent mean (mu)
        for i, layer_size in enumerate(price_head_layer_sizes):
            price_mean_modules.append(nn.Linear(current_dim_price_head, layer_size))
            price_logvar_modules.append(nn.Linear(current_dim_price_head, layer_size))
            if self.use_batch_norm: # Can use BN in price head too
                price_mean_modules.append(nn.BatchNorm1d(layer_size))
                price_logvar_modules.append(nn.BatchNorm1d(layer_size))
            price_mean_modules.append(activation_fn)
            price_logvar_modules.append(activation_fn)
            current_dim_price_head = layer_size

        self.price_mean_output = nn.Linear(current_dim_price_head, y_dim)
        self.price_logvar_output = nn.Linear(current_dim_price_head, y_dim) # Predicts log-variance of y

        price_mean_modules.append(self.price_mean_output)
        price_logvar_modules.append(self.price_logvar_output)

        self.price_mean_head = nn.Sequential(*price_mean_modules)
        self.price_logvar_head = nn.Sequential(*price_logvar_modules)

        self.logger.debug(f"Price Mean Head (for Y mean): {self.price_mean_head}")
        self.logger.debug(f"Price LogVar Head (for Y logvar): {self.price_logvar_head}")
        self.to(self.device_) # Ensure sub-modules are also moved
        self.logger.info(f"SemiSupMIWAE Initialized: InputXDim={input_dim}, TargetYDim={y_dim}, "
                         f"LatentDim={latent_dim}, AlphaPriceLoss={self.alpha_price_loss}, Device={self.device_}")


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        Returns: reconstruction_x, mu_latent, logvar_latent, y_pred_mu, y_pred_logvar
        """
        mu_latent, logvar_latent = self.encode(x)
        z = self.reparameterize(mu_latent, logvar_latent)
        reconstruction_x = self.decode_x(z) # Reconstruct X features

        # Price prediction heads take latent mu as input
        y_pred_mu = self.price_mean_head(mu_latent)
        y_pred_logvar = self.price_logvar_head(mu_latent)

        return reconstruction_x, mu_latent, logvar_latent, y_pred_mu, y_pred_logvar

    def predict_price(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts price (y_pred_mu) and its log-variance (y_pred_logvar) from input X.
        For inference/prediction time.
        """
        self.eval() # Ensure model is in evaluation mode
        with torch.no_grad():
            mu_latent, _ = self.encode(x) # Only need mu_latent for price prediction
            y_pred_mu = self.price_mean_head(mu_latent)
            y_pred_logvar = self.price_logvar_head(mu_latent)
        return y_pred_mu, y_pred_logvar

