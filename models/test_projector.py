import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Utilities ---------
def ensure_bct(x: torch.Tensor, mel_bins: int) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"mel must be 3D, got {x.shape}")
    if x.shape[1] == mel_bins:
        return x  # [B, 80, T]
    if x.shape[2] == mel_bins:
        return x.transpose(1, 2).contiguous()  # [B, 80, T]
    raise ValueError(f"Expected mel bins on dim 1 or 2, got {x.shape}")

def sinusoidal_pos_emb(T: int, dim: int, device, dtype):
    pe = torch.zeros(T, dim, device=device, dtype=dtype)
    pos = torch.arange(0, T, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # [T, dim]

# --------- Blocks ---------
class DSConvBlock(nn.Module):
    # Depthwise-separable 1D conv with residual and GLU gating
    def __init__(self, dim, kernel=7, dilation=1, dropout=0.0):
        super().__init__()
        pad = dilation * (kernel // 2)
        self.pw1 = nn.Conv1d(dim, dim*2, kernel_size=1)
        self.dw  = nn.Conv1d(dim*2, dim*2, kernel_size=kernel, padding=pad, dilation=dilation, groups=dim*2)
        self.act = nn.GLU(dim=1)
        self.pw2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.bn  = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # [B, C, T]
        residual = x
        x = self.pw1(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.drop(x)
        return self.bn(x + residual)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * d_model), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # [B, T, D]
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

# --------- Merged encoder + projector ---------
class MiniAudioEncoderProjector(nn.Module):
    """
    Input:  mel [B, T, 80] or [B, 80, T]
    Output:
      - projector='linear' -> [B, T', llm_dim], T' = ceil(T / stride)
      - projector='query'  -> [B, Nq, llm_dim]
    """
    def __init__(
        self,
        mel_bins=80,
        d_model=384,
        stride=8,
        conv_layers=2,
        tfm_layers=2,
        nhead=6,
        llm_dim=3072,
        projector='linear',      # 'linear' or 'query'
        num_queries=32,          # used if projector='query'
        dropout=0.1,
        use_input_ln=True,
        add_posenc=True,
    ):
        super().__init__()
        assert projector in ('linear', 'query')
        self.mel_bins = mel_bins
        self.d_model = d_model
        self.stride = max(1, int(stride))
        self.projector = projector
        self.add_posenc = add_posenc

        # Frontend
        self.in_norm = nn.LayerNorm(mel_bins) if use_input_ln else nn.Identity()
        self.proj_in = nn.Linear(mel_bins, d_model)
        self.pool = nn.AvgPool1d(kernel_size=self.stride, stride=self.stride, ceil_mode=True)
        self.convs = nn.ModuleList([DSConvBlock(d_model, kernel=7, dilation=2**i, dropout=dropout) for i in range(conv_layers)])
        self.tfms  = nn.ModuleList([TransformerBlock(d_model, nhead=nhead, mlp_ratio=4.0, dropout=dropout) for _ in range(tfm_layers)])
        self.pe_scale = nn.Parameter(torch.tensor(1.0))

        # Heads
        if projector == 'linear':
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, llm_dim))
        else:
            self.query = nn.Parameter(torch.randn(num_queries, d_model) / math.sqrt(d_model))
            self.q_ln  = nn.LayerNorm(d_model)
            self.cross = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout) for _ in range(2)])
            self.selfa = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout) for _ in range(2)])
            self.ffn   = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
                for _ in range(2)
            ])
            self.out   = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, llm_dim))

        # Initialize linears with Xavier for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        mel = ensure_bct(mel, self.mel_bins)                   # [B, 80, T]
        x = mel.transpose(1, 2).contiguous()                   # [B, T, 80]
        x = self.in_norm(x)
        x = self.proj_in(x)                                    # [B, T, D]
        x = x.transpose(1, 2)                                  # [B, D, T]
        x = self.pool(x)                                       # [B, D, T']
        for blk in self.convs:
            x = blk(x)                                         # [B, D, T']
        x = x.transpose(1, 2).contiguous()                     # [B, T', D]
        if self.add_posenc:
            pe = sinusoidal_pos_emb(x.size(1), x.size(2), x.device, x.dtype) * self.pe_scale
            x = x + pe
        for blk in self.tfms:
            x = blk(x)                                         # [B, T', D]
        return x

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.encode(mel)                                    # [B, T', D]
        if self.projector == 'linear':
            return self.head(x)                                 # [B, T', llm_dim]
        B = x.size(0)
        q = self.query.unsqueeze(0).expand(B, -1, -1)           # [B, Nq, D]
        q = self.q_ln(q)
        for ca, sa, ff in zip(self.cross, self.selfa, self.ffn):
            q = q + ca(q, x, x, need_weights=False)[0]
            q = q + sa(q, q, q, need_weights=False)[0]
            q = q + ff(q)
        return self.out(q)                                      # [B, Nq, llm_dim]

    # Helpers for testing
    def count_tokens(self, T: int) -> int:
        return (T + self.stride - 1) // self.stride  # ceil(T/stride)

    def tokens_per_second(self, frames_per_second: int) -> int:
        return (frames_per_second + self.stride - 1) // self.stride


if __name__ == "__main__":
    # Variable-rate
    m_var = MiniAudioEncoderProjector(projector='linear', stride=8, llm_dim=3072)
    x = torch.randn(2, 1000, 80)           # [B, T, 80]
    y = m_var(x)   
    print(y.shape)                         # [B, T', 3072]