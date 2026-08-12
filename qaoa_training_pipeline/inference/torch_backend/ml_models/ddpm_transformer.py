import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .edge_transformer import EdgeTransformer


class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        """
        Diffusion schedule for the DDPM process
        """
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def get_noise_params(self, t):
        # Ensure all tensors are on the same device as t
        # Move self tensors to the same device as t
        sqrt_alphas_t = self.sqrt_alphas_cumprod.to(t.device)[t]
        sqrt_one_minus_alphas_t = self.sqrt_one_minus_alphas_cumprod.to(t.device)[t]
        return sqrt_alphas_t, sqrt_one_minus_alphas_t


class DDPMTransformer(nn.Module):
    """
    DDPM Transformer using EdgeTransformer as the encoder.
    This model combines diffusion processes with graph-structured attention.
    The EdgeTransformer processes both node features and edge information.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 1e-3,
        edge_embed_dim: int = 32,
        max_nodes: int = None,  # Deprecated: kept for backward compatibility, not used
        permute_nodes_during_training: bool = True,
    ):
        super(DDPMTransformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.timesteps = timesteps

        # Diffusion schedule
        self.diffusion_schedule = DiffusionSchedule(
            timesteps=timesteps, beta_start=beta_start, beta_end=beta_end
        )

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Edge Transformer encoder with positional encoding for variable graph sizes
        self.transformer_encoder = EdgeTransformer(
            input_dim=embed_dim,
            output_dim=embed_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            edge_embed_dim=edge_embed_dim,
            use_positional_encoding=True,  # Use sinusoidal encoding for variable graph sizes
            permute_nodes_during_training=permute_nodes_during_training,
        )

        # Output projection
        self.output_proj = nn.Linear(embed_dim, output_dim)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edges, t, edge_weights=None):
        """
        x: (B, D) or (B, N, D)
        t: (B,) int timesteps or scalar
        edges: graph connectivity/features (clean), required
        edge_weights: (B, num_edges) edge weights (optional)
        """
        if edges is None:
            raise ValueError("Edge tensor is required for EdgeTransformer")

        B = x.shape[0]
        device = x.device

        if t.ndim == 0:
            t = t.expand(B)
        elif t.ndim == 2:
            t = t[:, 0] if t.shape[0] == B else t.reshape(B, -1)[:, 0]
        elif t.ndim == 1 and t.numel() != B:
            t = t[:B]
        t = t.to(device).long()

        if self.training:
            # Diffusion-style input augmentation: add noise scaled by timestep t.
            sqrt_alpha_t, sqrt_oma_t = self.diffusion_schedule.get_noise_params(t)
            bshape = (B,) + (1,) * (x.ndim - 1)  # -> (B,1) or (B,1,1)
            sqrt_alpha_t = sqrt_alpha_t.view(*bshape)
            sqrt_oma_t = sqrt_oma_t.view(*bshape)
            noise = torch.randn_like(x)
            x_t = sqrt_alpha_t * x + sqrt_oma_t * noise
        else:
            # At eval/test the model is a one-shot regressor: use the clean input
            # (t=0, no noise) so predictions are deterministic
            t = torch.zeros_like(t)
            x_t = x

        # project to embed space
        x_emb = self.input_proj(x_t)  # (B, E) or (B, N, E)

        # time embedding in embed space
        t_emb = self.time_embedding(self._timestep_embedding(t, self.embed_dim))  # (B, E)
        if x_emb.ndim == 3:  # (B, N, E)
            t_emb = t_emb[:, None, :]  # broadcast across nodes

        x_emb = x_emb + t_emb

        # encode with EdgeTransformer in embed space
        h = self.transformer_encoder(x_emb, edges, edge_weights)  # (B, E) or (B, N, E)

        # head
        h = self.layer_norm(h)
        h = self.dropout(h)
        out = self.output_proj(h)  # (B, output_dim) or (B, N, output_dim)
        return out

    def _timestep_embedding(self, timesteps, dim):
        """
        Create sinusoidal timestep embeddings
        """
        half = dim // 2
        emb = math.log(10000) / (half - 1)
        # Ensure all tensors are on the same device as timesteps
        device = timesteps.device
        freq_emb = torch.exp(torch.arange(half, dtype=torch.float, device=device) * -emb)
        emb = timesteps.float()[:, None] * freq_emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode="constant", value=0)
        return emb.to(device)
