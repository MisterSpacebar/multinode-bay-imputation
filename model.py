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
        return out


class SpatialGAT(nn.Module):
    """Stack of SpatialAttentionLayers."""

    def __init__(self, d_model: int, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [SpatialAttentionLayer(d_model, d_model) for _ in range(n_layers)]
        )

    def forward(self, h, edge_index, edge_weight):
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
        return h


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
    """

    def __init__(self, n_features: int, n_nodes: int,
                 d_model: int = 64,
                 n_gat_layers: int = 3,
                 n_gru_layers: int = 2):
        super().__init__()
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.d_model = d_model

        # Input dim:  features + mask_flags + rain (1) + sin_hour + cos_hour
        in_dim = n_features + n_features + 1 + 2

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
        x: torch.Tensor,           # (B, T, N, F) or (T, N, F)
        mask: torch.Tensor,        # same shape as x
        rain: torch.Tensor,        # (B, T) or (T,)
        edge_index: torch.Tensor,  # (2, E)
        edge_weight: torch.Tensor, # (E,)
        timestamps: torch.Tensor,  # (B, T) or (T,)  — unix seconds
    ):
        # Normalise to always work with a batch dimension
        unbatched = x.dim() == 3
        if unbatched:
            x          = x.unsqueeze(0)
            mask       = mask.unsqueeze(0)
            rain       = rain.unsqueeze(0)
            timestamps = timestamps.unsqueeze(0)

        B, T, N, F = x.shape

        x_filled = torch.where(mask.bool(), x, torch.zeros_like(x))

        # Time encodings: (B, T, 2)
        t_enc = self.time_encodings(timestamps.reshape(-1)).view(B, T, 2)

        # Build node features (B, T, N, in_dim)
        rain_exp = rain.view(B, T, 1, 1).expand(B, T, N, 1)
        t_exp    = t_enc.view(B, T, 1, 2).expand(B, T, N, 2)
        node_in  = torch.cat([x_filled, mask.float(), rain_exp, t_exp], dim=-1)

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
        h_spatial = self.spatial(h_flat, ei_t, ew_t).view(B, T, N, self.d_model)

        # 3. Temporal GRU — (B*N, T, d)
        h_t   = h_spatial.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
        h_rnn, _ = self.gru(h_t)
        h_rnn = h_rnn.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)  # (B,T,N,d)

        # 4. Decode
        pred = self.decoder(h_rnn)   # (B, T, N, F)

        imputed = torch.where(mask.bool(), x_filled, pred)

        if unbatched:
            return imputed.squeeze(0), pred.squeeze(0)
        return imputed, pred


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
