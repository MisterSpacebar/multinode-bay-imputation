"""
model.py
--------
Spatio-Temporal Graph Neural Network for multi-node sensor imputation.

Architecture (one forward pass over a window of T timesteps):

  ┌──────────────────────────────────────────────────────────┐
  │  For each timestep t:                                     │
  │                                                           │
  │  1. Node Encoder                                          │
  │     [sensor_feats | rain | sin/cos time | mask_flags]     │
  │       → Linear → LayerNorm → ReLU → h_enc  (N, d_model)  │
  │                                                           │
  │  2. Graph Attention (GATConv × n_layers)                  │
  │     h_enc + edge_index + edge_weight                      │
  │       → h_graph  (N, d_model)                             │
  │     Each node attends to its k nearest neighbours;        │
  │     attention scores are modulated by edge_weight         │
  │                                                           │
  │  3. Temporal GRU                                          │
  │     h_graph, hidden_t-1  →  h_rnn, hidden_t  (N, d_model)│
  │                                                           │
  │  4. Decoder                                               │
  │     h_rnn  →  Linear → reconstructed sensor values (N, F)│
  │                                                           │
  │  Imputation rule:                                         │
  │     output[t] = known_values[t]  where mask[t] == 1       │
  │     output[t] = model_pred[t]    where mask[t] == 0       │
  └──────────────────────────────────────────────────────────┘

Training uses random masking so the model learns to reconstruct any
subset of missing values from the spatial + temporal context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lightweight Graph Attention layer (no PyG dependency)
# ---------------------------------------------------------------------------

class SpatialAttentionLayer(nn.Module):
    """
    Single-head graph attention over a fixed adjacency given as edge_index.

    edge_index : (2, E) — [source_row, target_row]
    edge_weight: (E,)   — prior distance-based weights (used as bias)
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)
        self.scale = d_out ** -0.5
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        h           : (M, d_in)   M = any flattened batch*time*node combo
        edge_index  : (2, E_tot)  already tiled for all timesteps if needed
        edge_weight : (E_tot,)
        returns     : (M, d_out)
        """
        M = h.size(0)
        src, dst = edge_index[0], edge_index[1]

        Q = self.W_q(h[dst])
        K = self.W_k(h[src])
        V = self.W_v(h[src])

        attn = (Q * K).sum(-1) * self.scale + edge_weight

        # Per-destination softmax
        attn_exp = torch.zeros(M, device=h.device, dtype=h.dtype)
        attn_exp.scatter_add_(0, dst, attn.exp())
        attn_norm = attn.exp() / (attn_exp[dst] + 1e-9)

        out = torch.zeros(M, V.size(-1), device=h.device, dtype=h.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(V),
                         attn_norm.unsqueeze(-1) * V)

        if h.size(-1) == out.size(-1):
            out = self.norm(out + h)
        else:
            out = self.norm(out)
        return out, attn_norm   # always return weights; callers ignore if unneeded


class SpatialGAT(nn.Module):
    """Stack of SpatialAttentionLayers."""

    def __init__(self, d_model: int, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [SpatialAttentionLayer(d_model, d_model) for _ in range(n_layers)]
        )

    def forward(self, h, edge_index, edge_weight, collect_attn=False):
        attn_list = []
        for layer in self.layers:
            h, attn = layer(h, edge_index, edge_weight)
            if collect_attn:
                attn_list.append(attn)
        return h, attn_list


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class BayImputationGNN(nn.Module):
    """
    Parameters
    ----------
    n_features  : number of sensor channels (F)
    n_nodes     : number of stations (N)
    d_model     : hidden dimension
    n_gat_layers: depth of spatial attention stack
    n_gru_layers: GRU depth
    n_forcing   : number of external forcing scalars (rain, temp_min, temp_max)
    """

    def __init__(self, n_features: int, n_nodes: int,
                 d_model: int = 64,
                 n_gat_layers: int = 3,
                 n_gru_layers: int = 2,
                 n_forcing: int = 3):
        super().__init__()
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.n_forcing = n_forcing

        # Input dim: features + mask_flags + forcing scalars + sin/cos time
        in_dim = n_features + n_features + n_forcing + 2

        # 1. Node encoder (applied independently per node)
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # 2. Spatial GAT
        self.spatial = SpatialGAT(d_model, n_layers=n_gat_layers)

        # 3. Temporal GRU (processes nodes independently but shares weights)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_gru_layers,
            batch_first=True,
        )

        # 4. Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_features),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def time_encodings(timestamps_unix: torch.Tensor) -> torch.Tensor:
        """
        Circular encoding of time-of-day.
        timestamps_unix : (T,) float  — seconds since epoch
        returns         : (T, 2)      — [sin_hour, cos_hour]
        """
        seconds_in_day = 24.0 * 3600.0
        phase = (timestamps_unix % seconds_in_day) / seconds_in_day * 2 * torch.pi
        return torch.stack([phase.sin(), phase.cos()], dim=-1)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,            # (B, T, N, F) or (T, N, F)
        mask: torch.Tensor,         # same shape as x
        forcing: torch.Tensor,      # (B, T, n_forcing) or (T, n_forcing)
        edge_index: torch.Tensor,   # (2, E)
        edge_weight: torch.Tensor,  # (E,)
        timestamps: torch.Tensor,   # (B, T) or (T,)
        return_attention: bool = False,
    ):
        # Normalise to always work with a batch dimension
        unbatched = x.dim() == 3
        if unbatched:
            x          = x.unsqueeze(0)
            mask       = mask.unsqueeze(0)
            forcing    = forcing.unsqueeze(0)
            timestamps = timestamps.unsqueeze(0)

        B, T, N, F = x.shape

        x_filled = torch.where(mask.bool(), x, torch.zeros_like(x))

        # Time encodings: (B, T, 2)
        t_enc = self.time_encodings(timestamps.reshape(-1)).view(B, T, 2)

        # Build node features (B, T, N, in_dim)
        # forcing: (B, T, n_forcing) -> (B, T, N, n_forcing)
        forcing_exp = forcing.unsqueeze(2).expand(B, T, N, self.n_forcing)
        t_exp       = t_enc.view(B, T, 1, 2).expand(B, T, N, 2)
        node_in     = torch.cat([x_filled, mask.float(), forcing_exp, t_exp], dim=-1)

        # 1. Node encoder — flatten all dims for shared linear
        h_enc = self.node_encoder(node_in.view(B * T * N, -1))
        h_enc = h_enc.view(B * T, N, self.d_model)   # (B*T, N, d)

        # 2. Spatial GAT — tile edge graph over B*T "frames"
        BT = B * T
        E  = edge_index.shape[1]
        offsets = torch.arange(BT, device=x.device) * N          # (B*T,)
        src_t = (edge_index[0].unsqueeze(0) + offsets.unsqueeze(1)).view(-1)  # (B*T*E,)
        dst_t = (edge_index[1].unsqueeze(0) + offsets.unsqueeze(1)).view(-1)
        ew_t  = edge_weight.unsqueeze(0).expand(BT, -1).reshape(-1)
        ei_t  = torch.stack([src_t, dst_t], dim=0)

        h_flat    = h_enc.view(BT * N, self.d_model)
        h_spatial_flat, attn_list = self.spatial(
            h_flat, ei_t, ew_t, collect_attn=return_attention
        )
        h_spatial = h_spatial_flat.view(B, T, N, self.d_model)

        # 3. Temporal GRU — (B*N, T, d)
        h_t   = h_spatial.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
        h_rnn, _ = self.gru(h_t)
        h_rnn = h_rnn.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)  # (B,T,N,d)

        # 4. Decode
        pred = self.decoder(h_rnn)   # (B, T, N, F)

        imputed = torch.where(mask.bool(), x_filled, pred)

        # Build (N, N) attention matrix averaged over layers and B*T frames
        attn_matrix = None
        if return_attention and attn_list:
            # attn_list: list of (B*T*E,) tensors, one per GAT layer
            # Average across layers → (B*T*E,), then reshape to (B*T, E) → mean over frames → (E,)
            avg = torch.stack([a for a in attn_list], dim=0).mean(0)  # (B*T*E,)
            avg = avg.view(BT, E).mean(0)                              # (E,)
            # Scatter to (N, N)
            orig_src = edge_index[0]  # (E,) — un-tiled
            orig_dst = edge_index[1]
            mat = torch.zeros(N, N, device=x.device)
            for k in range(E):
                mat[orig_dst[k], orig_src[k]] = avg[k]
            attn_matrix = mat

        if unbatched:
            return imputed.squeeze(0), pred.squeeze(0), attn_matrix
        return imputed, pred, attn_matrix


# ---------------------------------------------------------------------------
# Normaliser — stored alongside the model so inference uses the same stats
# ---------------------------------------------------------------------------

class Normaliser:
    """Per-feature (across all nodes and time) z-score normalisation."""

    def __init__(self):
        self.mean = None   # (F,)
        self.std = None    # (F,)

    def fit(self, X: "np.ndarray"):
        """X : (T, N, F)  — fit ignores NaNs."""
        import numpy as np
        T, N, F = X.shape
        flat = X.reshape(-1, F)
        self.mean = np.nanmean(flat, axis=0).astype(np.float32)
        self.std  = np.nanstd(flat,  axis=0).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0   # avoid divide-by-zero

    def transform(self, X: "np.ndarray") -> "np.ndarray":
        return (X - self.mean) / self.std

    def inverse_transform(self, X: "np.ndarray") -> "np.ndarray":
        return X * self.std + self.mean

    def to_tensors(self, device="cpu"):
        import torch
        return (torch.tensor(self.mean, device=device),
                torch.tensor(self.std,  device=device))
