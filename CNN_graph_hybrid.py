import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import classification_report, accuracy_score
from scipy.spatial import cKDTree
from collections import Counter
import time

# ── 0. Device ─────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── 1. Config ─────────────────────────────────────────────────────────
DATA_DIR   = '/workspace/Data-Flood/'
PATCH_SIZE = 11
HALF       = PATCH_SIZE // 2
BATCH_SIZE = 512
N_EPOCHS = 60
LR       = 1e-3
TARGET_COL = 'risk_0_2m'
BASIN_SIZE = 20   # 50x50 pixels = 1km x 1km per node

FEATURE_COLS = [
    'dtm_zscore', 'log_flow_acc', 'imd', 'waw',
    'is_waterway', 'clc_type_clean',
    'tp_p99_zscore', 'max_rolling5_tp_zscore',
    'sro_p95_zscore', 'swvl1_min_zscore'
]
N_CHANNELS = len(FEATURE_COLS)
print(f"Features ({N_CHANNELS}): {FEATURE_COLS}")

# ── 2. Load data ──────────────────────────────────────────────────────
print("\nLoading datasets...")
ds_terrain_severn      = xr.open_dataset(DATA_DIR + 'Copy of Copy of flood_risk_terrain_severn.nc',      engine='netcdf4')
ds_terrain_northumbria = xr.open_dataset(DATA_DIR + 'Copy of Copy of flood_risk_terrain_northumbria.nc', engine='netcdf4')
ds_era5_severn         = xr.open_dataset(DATA_DIR + 'Copy of Copy of era5_land_severn.nc',               engine='netcdf4')
ds_era5_northumbria    = xr.open_dataset(DATA_DIR + 'Copy of Copy of era5_land_northumbria.nc',           engine='netcdf4')
print("All datasets loaded.")

# ── 3. Feature engineering ────────────────────────────────────────────
risk_cols = ['risk_0_2m','risk_0_3m','risk_0_6m','risk_0_9m','risk_1_2m']

def filter_risk_pixels(df):
    mask = df[risk_cols].notna().any(axis=1) & (df[risk_cols] != 0).any(axis=1)
    return df[mask].copy()

def engineer_terrain(df):
    df = df.copy()
    df['waw']            = df['waw'].where(df['waw'] <= 4, np.nan).fillna(0)
    df['imd']            = df['imd'].where(df['imd'] <= 100, np.nan).fillna(0)
    df['dtm_m']          = df['dtm'] / 10
    df['dtm_zscore']     = (df['dtm_m'] - df['dtm_m'].mean()) / df['dtm_m'].std()
    df['log_flow_acc']   = np.log1p(df['flow_acc'])
    df['is_waterway']    = df['rciw'].notna().astype(int)
    df['clc_type_clean'] = df['clc_type'].fillna(df['clc_type'].median())
    return df

def engineer_weather(ds):
    df_w = ds[['tp','sro','swvl1_mean','swvl1_max']].to_dataframe().reset_index()
    df_w = df_w.dropna(subset=['tp'])
    grouped = df_w.groupby(['y','x'])
    w = grouped.agg(
        tp_p99    = ('tp',         lambda x: np.percentile(x, 99)),
        sro_p95   = ('sro',        lambda x: np.percentile(x, 95)),
        swvl1_min = ('swvl1_mean', 'min'),
    ).reset_index()
    df_w_s = df_w.sort_values(['y','x','valid_time'])
    df_w_s['tp_r5'] = (
        df_w_s.groupby(['y','x'])['tp']
        .transform(lambda x: x.rolling(5, min_periods=5).sum())
    )
    r = df_w_s.groupby(['y','x']).agg(
        max_rolling5_tp=('tp_r5','max')
    ).reset_index()
    return w.merge(r, on=['y','x'])

def merge_weather(df_t, df_w):
    tree = cKDTree(df_w[['y','x']].values)
    _, idx = tree.query(df_t[['y','x']].values, k=1)
    cols = [c for c in df_w.columns if c not in ['y','x']]
    matched = df_w.iloc[idx][cols].reset_index(drop=True)
    return pd.concat([df_t.reset_index(drop=True), matched], axis=1)

def add_weather_zscore(df, cols):
    df = df.copy()
    for c in cols:
        m, s = df[c].mean(), df[c].std()
        df[f'{c}_zscore'] = (df[c] - m) / (s + 1e-8)
    return df

print("Processing Severn...")
df_s = ds_terrain_severn.to_dataframe().reset_index()
df_s = filter_risk_pixels(df_s)
df_s = df_s[df_s['risk_0_2m'].isin([1.,2.,3.,4.])].copy()
df_s = engineer_terrain(df_s)
w_s  = engineer_weather(ds_era5_severn)
df_s = merge_weather(df_s, w_s)
df_s = add_weather_zscore(df_s, ['tp_p99','sro_p95','swvl1_min','max_rolling5_tp'])
print(f"Severn: {len(df_s):,} pixels")

print("Processing Northumbria...")
df_n = ds_terrain_northumbria.to_dataframe().reset_index()
df_n = filter_risk_pixels(df_n)
df_n = engineer_terrain(df_n)
w_n  = engineer_weather(ds_era5_northumbria)
df_n = merge_weather(df_n, w_n)
df_n = add_weather_zscore(df_n, ['tp_p99','sro_p95','swvl1_min','max_rolling5_tp'])
print(f"Northumbria: {len(df_n):,} pixels")

# ── 4. Build raster grid ──────────────────────────────────────────────
def df_to_grid(df, feature_cols, target_col, resolution=20):
    df = df.copy()
    df['yr'] = (df['y'] / resolution).round().astype(int)
    df['xr'] = (df['x'] / resolution).round().astype(int)
    df = df.drop_duplicates(subset=['yr','xr'])
    yr_vals = np.sort(df['yr'].unique())[::-1]
    xr_vals = np.sort(df['xr'].unique())
    yr_to_i = {v: i for i, v in enumerate(yr_vals)}
    xr_to_j = {v: j for j, v in enumerate(xr_vals)}
    H, W, C = len(yr_vals), len(xr_vals), len(feature_cols)
    print(f"  Grid: {H} × {W} = {H*W:,} cells | {C} channels")
    feat_grid  = np.zeros((H, W, C), dtype=np.float32)
    label_grid = np.full((H, W), -1, dtype=np.int8)
    flow_grid  = np.zeros((H, W), dtype=np.float32)
    feat_vals  = df[feature_cols].fillna(0).values.astype(np.float32)
    labels     = df[target_col].values
    fdir       = df['flow_dir'].fillna(0).values
    yr_idx     = df['yr'].map(yr_to_i).values
    xr_idx     = df['xr'].map(xr_to_j).values
    for k in range(len(df)):
        i, j = yr_idx[k], xr_idx[k]
        feat_grid[i, j, :]  = feat_vals[k]
        flow_grid[i, j]      = fdir[k]
        lbl = labels[k]
        if not np.isnan(lbl) and lbl in [1,2,3,4]:
            label_grid[i, j] = int(lbl) - 1
    return feat_grid, label_grid, flow_grid

print("\nBuilding Severn grid...")
df_s_clean = df_s.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
grid_s, labels_s, fdir_s = df_to_grid(df_s_clean, FEATURE_COLS, TARGET_COL)

print("Building Northumbria grid...")
df_n_clean = df_n.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
grid_n, labels_n, fdir_n = df_to_grid(df_n_clean, FEATURE_COLS, TARGET_COL)

# ── 5. Build hydrological basin graph (pure PyTorch, no torch-geometric)
def build_basin_graph(feat_grid, label_grid, flow_dir_grid, basin_size=50):
    H, W, C = feat_grid.shape
    bs      = basin_size
    n_rows  = H // bs
    n_cols  = W // bs
    n_nodes = n_rows * n_cols

    print(f"  Basin grid: {n_rows} × {n_cols} = {n_nodes} nodes")
    print(f"  Each node: {bs*20}m × {bs*20}m = {bs*20/1000:.1f}km²")

    node_feats  = np.zeros((n_nodes, C), dtype=np.float32)
    node_labels = np.full(n_nodes, -1, dtype=np.int8)
    node_valid  = np.zeros(n_nodes, dtype=bool)
    node_flow   = np.zeros(n_nodes, dtype=np.float32)

    for bi in range(n_rows):
        for bj in range(n_cols):
            nid  = bi * n_cols + bj
            r0, r1 = bi*bs, min((bi+1)*bs, H)
            c0, c1 = bj*bs, min((bj+1)*bs, W)

            patch_feats  = feat_grid[r0:r1, c0:c1, :]
            patch_labels = label_grid[r0:r1, c0:c1]
            patch_flow   = flow_dir_grid[r0:r1, c0:c1]

            node_feats[nid] = patch_feats.reshape(-1, C).mean(axis=0)
            node_flow[nid]  = patch_flow.mean()

            valid_px = patch_labels[patch_labels >= 0]
            if len(valid_px) > 0:
                counts = np.bincount(valid_px, minlength=4)
                node_labels[nid] = counts.argmax()
                node_valid[nid]  = True

    # Build edges
    def flow_to_offset(angle_deg):
        a = angle_deg % 360
        if   a < 22.5  or a >= 337.5: return ( 0,  1)
        elif a < 67.5:                 return (-1,  1)
        elif a < 112.5:                return (-1,  0)
        elif a < 157.5:                return (-1, -1)
        elif a < 202.5:                return ( 0, -1)
        elif a < 247.5:                return ( 1, -1)
        elif a < 292.5:                return ( 1,  0)
        else:                          return ( 1,  1)

    edge_src, edge_dst, edge_attr = [], [], []

    for bi in range(n_rows):
        for bj in range(n_cols):
            src = bi * n_cols + bj
            if not node_valid[src]:
                continue
            # Spatial adjacency edges
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = bi+di, bj+dj
                    if 0 <= ni < n_rows and 0 <= nj < n_cols:
                        dst = ni * n_cols + nj
                        if node_valid[dst]:
                            edge_src.append(src)
                            edge_dst.append(dst)
                            edge_attr.append(0.0)  # spatial
            # Hydrological flow edge
            dr, dc = flow_to_offset(node_flow[src])
            ni, nj = bi+dr, bj+dc
            if 0 <= ni < n_rows and 0 <= nj < n_cols:
                dst = ni * n_cols + nj
                if node_valid[dst] and dst != src:
                    edge_src.append(src)
                    edge_dst.append(dst)
                    edge_attr.append(1.0)  # hydrological

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    # After building edge_src, edge_dst, edge_attr lists
    # Filter to only keep edges where both endpoints are valid
    valid_set = set(np.where(node_valid)[0].tolist())
    filtered_src, filtered_dst, filtered_attr = [], [], []
    for s, d, a in zip(edge_src, edge_dst, edge_attr):
        if s in valid_set and d in valid_set:
            filtered_src.append(s)
            filtered_dst.append(d)
            filtered_attr.append(a)

    edge_index = torch.tensor([filtered_src, filtered_dst], dtype=torch.long)
    edge_attr  = torch.tensor(filtered_attr, dtype=torch.float).unsqueeze(1)
    edge_attr  = torch.tensor(edge_attr,  dtype=torch.float).unsqueeze(1)
    x          = torch.tensor(node_feats, dtype=torch.float)
    y          = torch.tensor(node_labels,dtype=torch.long)
    valid_mask = torch.tensor(node_valid, dtype=torch.bool)

    n_spatial = sum(1 for e in edge_attr if e.item() == 0)
    n_flow    = sum(1 for e in edge_attr if e.item() == 1)
    print(f"  Valid nodes: {node_valid.sum():,}")
    print(f"  Edges — spatial: {n_spatial:,} | flow: {n_flow:,} | total: {len(edge_src):,}")

    return x, edge_index, edge_attr, y, valid_mask, n_rows, n_cols

print("\nBuilding Severn basin graph...")
x_s, ei_s, ea_s, y_s, vm_s, nr_s, nc_s = build_basin_graph(grid_s, labels_s, fdir_s, BASIN_SIZE)

print("\nBuilding Northumbria basin graph...")
x_n, ei_n, ea_n, y_n, vm_n, nr_n, nc_n = build_basin_graph(grid_n, labels_n, fdir_n, BASIN_SIZE)

# ── 6. Extract basin patches for CNN ─────────────────────────────────
def extract_basin_patches(feat_grid, n_rows, n_cols, basin_size):
    H, W, C = feat_grid.shape
    bs      = basin_size
    patches = []
    for bi in range(n_rows):
        for bj in range(n_cols):
            r0, c0 = bi*bs, bj*bs
            r1, c1 = min(r0+bs, H), min(c0+bs, W)
            patch  = feat_grid[r0:r1, c0:c1, :]
            if patch.shape[0] < bs or patch.shape[1] < bs:
                pad = np.zeros((bs, bs, C), dtype=np.float32)
                pad[:patch.shape[0], :patch.shape[1], :] = patch
                patch = pad
            patches.append(patch.transpose(2, 0, 1))  # (C, bs, bs)
    return torch.tensor(np.stack(patches), dtype=torch.float)

print("\nExtracting basin patches...")
patches_s = extract_basin_patches(grid_s, nr_s, nc_s, BASIN_SIZE)
patches_n = extract_basin_patches(grid_n, nr_n, nc_n, BASIN_SIZE)
print(f"Severn patches: {patches_s.shape}")
print(f"Northumbria patches: {patches_n.shape}")

# ── 7. Manual GAT (no torch-geometric dependency) ────────────────────
class ManualGATConv(nn.Module):
    """Single-head GAT — no shape ambiguity."""
    def __init__(self, in_dim, out_dim, heads=4, edge_dim=1, dropout=0.2):
        super().__init__()
        # ignore heads — single head for stability
        self.out_dim = out_dim
        self.W_src   = nn.Linear(in_dim,   out_dim, bias=False)
        self.W_dst   = nn.Linear(in_dim,   out_dim, bias=False)
        self.W_edge  = nn.Linear(edge_dim, 1,       bias=False)
        self.a       = nn.Linear(out_dim,  1,       bias=False)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        src_idx, dst_idx = edge_index[0], edge_index[1]
        N = x.size(0)
        E = src_idx.size(0)

        # (N, out_dim)
        h_src = self.W_src(x)
        h_dst = self.W_dst(x)

        # (E, out_dim)
        e_src = h_src[src_idx]
        e_dst = h_dst[dst_idx]

        # attention score (E, 1)
        e_feat = self.W_edge(edge_attr.float())          # (E, 1)
        score  = self.a(torch.tanh(e_src + e_dst)).view(-1)   # force (E,)
        score  = score + e_feat.view(-1)                       # force (E,)
        score  = torch.nn.functional.leaky_relu(score, 0.2)   # (E,)

        # softmax per destination
        score  = score - score.max()
        score  = torch.exp(score)                        # (E,)
        score  = self.drop(score)

        # normalize
        norm = torch.zeros(N, device=x.device, dtype=score.dtype)
        norm.scatter_add_(0, dst_idx.view(-1), score.view(-1))
        norm = norm.clamp(min=1e-6)
        alpha = score / norm[dst_idx]                    # (E,)

        # aggregate messages
        msgs = e_src * alpha.unsqueeze(1)                # (E, out_dim)
        out  = torch.zeros(N, self.out_dim,
                           device=x.device, dtype=msgs.dtype)
        idx = dst_idx.view(-1, 1).expand(E, self.out_dim)
        out.scatter_add_(0, idx, msgs)
        return out                                       # (N, out_dim)                       # (N, H*D)                                 # (N, H*D)                                  # (N, H*D)                             # (N, H*D)
# ── 8. GNN ────────────────────────────────────────────────────────────
class HydroGNN(nn.Module):
    def __init__(self, node_dim, hidden_dim=128, n_classes=4, n_layers=3):
        super().__init__()
        
        # hidden_dim must be divisible by heads
        # heads=4, so hidden_dim must be multiple of 4
        assert hidden_dim % 4 == 0, "hidden_dim must be divisible by 4"
        
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )
        
        # Each GAT layer: in=hidden_dim, out=hidden_dim, heads=4
        # out_dim per head = hidden_dim // 4
        self.gat_layers = nn.ModuleList([
    ManualGATConv(
        in_dim   = hidden_dim,
        out_dim  = hidden_dim,
        heads    = 1,          # ignored internally
        edge_dim = 1,
        dropout  = 0.2
    )
    for _ in range(n_layers)
])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.input_proj(x)              # (N, hidden_dim)
        
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            gat_out = gat(x, edge_index, edge_attr)  # (N, hidden_dim)
            x = ln(x + gat_out)                       # residual — same shape
        
        return self.classifier(x)

# ── 9. CNN patch encoder ──────────────────────────────────────────────
class LocalCNNExtractor(nn.Module):
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.proj(self.gap(self.encoder(x)))

# ── 10. Hybrid CNN-GNN ────────────────────────────────────────────────
class HybridCNNGNN(nn.Module):
    """
    Hierarchical hybrid model:
    - CNN extracts local terrain texture from each 1km² basin patch
    - GNN propagates flood risk signals along hydrological graph
    - Combines local + catchment-scale spatial understanding
    """
    def __init__(self, in_channels, cnn_dim=64, gnn_hidden=128, n_classes=4):
        super().__init__()
        self.cnn = LocalCNNExtractor(in_channels, out_dim=cnn_dim)
        self.gnn = HydroGNN(cnn_dim + in_channels, gnn_hidden, n_classes)

    def forward(self, patches, x_node, edge_index, edge_attr):
        cnn_feats  = self.cnn(patches)                           # (N, cnn_dim)
        node_feats = torch.cat([cnn_feats, x_node], dim=1)       # (N, cnn_dim+C)
        return self.gnn(node_feats, edge_index, edge_attr)        # (N, n_classes)

# ── 11. Train/val split ───────────────────────────────────────────────
valid_ids = vm_s.nonzero(as_tuple=True)[0].numpy()
rng       = np.random.default_rng(42)
shuffled  = rng.permutation(valid_ids)
n_val     = int(len(valid_ids) * 0.2)

val_mask   = torch.zeros(x_s.shape[0], dtype=torch.bool)
train_mask = torch.zeros(x_s.shape[0], dtype=torch.bool)
for nid in shuffled[:n_val]:
    val_mask[nid]   = True
for nid in shuffled[n_val:]:
    train_mask[nid] = True

print(f"\nTrain nodes: {train_mask.sum().item():,}")
print(f"Val nodes:   {val_mask.sum().item():,}")
print(f"Test nodes:  {vm_n.sum().item():,}")

# ── 12. Model setup ───────────────────────────────────────────────────
model = HybridCNNGNN(N_CHANNELS, cnn_dim=32, gnn_hidden=64, n_classes=4).to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Class weights
tr_labels = y_s[train_mask].numpy()
tr_labels = tr_labels[tr_labels >= 0]
cc        = Counter(tr_labels.tolist())
tot       = sum(cc.values())
raw_weights = [tot / (4 * cc.get(c, 1)) for c in range(4)]
# Cap at 2x to prevent collapse
max_w = 2.0
cw = torch.tensor(
    [min(w, max_w) for w in raw_weights],
    dtype=torch.float32
).to(device)
print(f"Class weights (capped): {cw.cpu().numpy().round(3)}")

criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
# Replace scheduler with warmup + cosine
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR,
    total_steps=N_EPOCHS,
    pct_start=0.1
)
scaler    = torch.amp.GradScaler('cuda')

# Move everything to device
patches_s_cpu  = patches_s
x_s_dev        = x_s.to(device)
ei_s_dev       = ei_s.to(device)
ea_s_dev       = ea_s.to(device)
y_s_dev        = y_s.to(device)
train_mask_dev = train_mask.to(device)
val_mask_dev   = val_mask.to(device)

patches_n_cpu  = patches_n
x_n_dev        = x_n.to(device)
ei_n_dev       = ei_n.to(device)
ea_n_dev       = ea_n.to(device)
y_n_dev        = y_n.to(device)
vm_n_dev       = vm_n.to(device)

# ── 13. Training loop ─────────────────────────────────────────────────
print(f"\nTraining for {N_EPOCHS} epochs on {device}...")
print(f"{'Epoch':>5} {'TrLoss':>8} {'TrAcc':>7} {'VlLoss':>8} {'VlAcc':>7} {'Time':>7}")
print("-" * 50)

best_val_acc = 0
best_state   = None

def forward_chunked(model, patches, x_node, edge_index, edge_attr,
                    device, chunk_size=128):
    cnn_feats = []
    for i in range(0, patches.shape[0], chunk_size):
        chunk = patches[i:i+chunk_size].to(device)
        feats = model.cnn(chunk)
        cnn_feats.append(feats)
        del chunk
    cnn_feats  = torch.cat(cnn_feats, dim=0)
    node_feats = torch.cat([cnn_feats, x_node], dim=1)
    return model.gnn(node_feats, edge_index, edge_attr)

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()

    # ── Train ──────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    with torch.amp.autocast('cuda'):
        logits = forward_chunked(model, patches_s_cpu, x_s_dev, ei_s_dev, ea_s_dev, device)
        tr_log = logits[train_mask_dev]
        tr_lbl = y_s_dev[train_mask_dev]
        valid  = tr_lbl >= 0
        loss   = criterion(tr_log[valid], tr_lbl[valid])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    tr_acc  = (tr_log[valid].argmax(1) == tr_lbl[valid]).float().mean().item()
    tr_loss = loss.item()

    # ── Validate ───────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        logits  = forward_chunked(model, patches_s_cpu, x_s_dev, ei_s_dev, ea_s_dev, device)

        vl_log  = logits[val_mask_dev]
        vl_lbl  = y_s_dev[val_mask_dev]
        valid_v = vl_lbl >= 0
        vl_loss = criterion(vl_log[valid_v], vl_lbl[valid_v]).item()
        vl_acc  = (vl_log[valid_v].argmax(1) == vl_lbl[valid_v]).float().mean().item()

    elapsed = time.time() - t0
    marker  = ' ◄' if vl_acc > best_val_acc else ''
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"{epoch:>5} {tr_loss:>8.4f} {tr_acc:>7.3f} "
          f"{vl_loss:>8.4f} {vl_acc:>7.3f} {elapsed:>6.1f}s{marker}")

# ── 14. Final evaluation ──────────────────────────────────────────────
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
model.eval()

with torch.no_grad(), torch.amp.autocast('cuda'):
    # Severn val
    logits  = forward_chunked(model, patches_s_cpu, x_s_dev, ei_s_dev, ea_s_dev, device)
    vl_log  = logits[val_mask_dev]
    vl_lbl  = y_s_dev[val_mask_dev]
    valid_v = vl_lbl >= 0
    vl_preds = vl_log[valid_v].argmax(1).cpu().numpy()
    vl_true  = vl_lbl[valid_v].cpu().numpy()

    # Northumbria test
    logits   = forward_chunked(model, patches_n_cpu, x_n_dev, ei_n_dev, ea_n_dev, device)
    ts_log   = logits[vm_n_dev]
    ts_lbl   = y_n_dev[vm_n_dev]
    valid_t  = ts_lbl >= 0
    ts_preds = ts_log[valid_t].argmax(1).cpu().numpy()
    ts_true  = ts_lbl[valid_t].cpu().numpy()

print("\n── Hybrid CNN-GNN: Severn val (seen) ──")
print(f"Accuracy: {accuracy_score(vl_true, vl_preds):.3f}")
print(classification_report(vl_true, vl_preds,
      target_names=['Very Low','Low','Medium','High']))

print("── Hybrid CNN-GNN: Northumbria test (unseen) ──")
print(f"Accuracy: {accuracy_score(ts_true, ts_preds):.3f}")
print(classification_report(ts_true, ts_preds,
      target_names=['Very Low','Low','Medium','High']))

# ── 15. Summary ───────────────────────────────────────────────────────
print("\n── Model progression summary ──")
print(f"XGBoost v4 (baseline)   — Severn: 0.582 | Northumbria: 0.427")
print(f"CNN v2                  — Severn: 0.504 | Northumbria: 0.362")
print(f"Hybrid CNN-GNN          — Severn: {accuracy_score(vl_true, vl_preds):.3f} | Northumbria: {accuracy_score(ts_true, ts_preds):.3f}")

torch.save(best_state, '/workspace/Flood-Risk/flood_hybrid_cnn_gnn.pt')
print("\nModel saved to /workspace/Flood-Risk/flood_hybrid_cnn_gnn.pt")