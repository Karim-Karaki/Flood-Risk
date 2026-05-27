import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch
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
DATA_DIR    = '/workspace/Data-Flood/'
PATCH_SIZE  = 11
HALF        = PATCH_SIZE // 2
BATCH_SIZE  = 512        # smaller — graph batches are memory heavy
N_EPOCHS    = 40
LR          = 3e-4
TARGET_COL  = 'risk_0_2m'

# Superpixel/basin config
# We partition the raster into spatial blocks (basins)
# Each block becomes a graph node
# Blocks are connected by hydrological flow relationships
BASIN_SIZE  = 50   # 50x50 pixel blocks = 1km x 1km basins at 20m resolution

FEATURE_COLS = [
    'dtm_zscore', 'log_flow_acc', 'imd', 'waw',
    'is_waterway', 'clc_type_clean',
    'tp_p99_zscore', 'max_rolling5_tp_zscore',
    'sro_p95_zscore', 'swvl1_min_zscore'
]
N_CHANNELS = len(FEATURE_COLS)
print(f"Features ({N_CHANNELS}): {FEATURE_COLS}")

# ── 2. Load + engineer features (reuse previous pipeline) ────────────
print("\nLoading datasets...")
ds_terrain_severn      = xr.open_dataset(DATA_DIR + 'Copy of Copy of flood_risk_terrain_severn.nc',      engine='netcdf4')
ds_terrain_northumbria = xr.open_dataset(DATA_DIR + 'Copy of Copy of flood_risk_terrain_northumbria.nc', engine='netcdf4')
ds_era5_severn         = xr.open_dataset(DATA_DIR + 'Copy of Copy of era5_land_severn.nc',               engine='netcdf4')
ds_era5_northumbria    = xr.open_dataset(DATA_DIR + 'Copy of Copy of era5_land_northumbria.nc',           engine='netcdf4')

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
    df_w_s['tp_r5'] = (df_w_s.groupby(['y','x'])['tp']
                       .transform(lambda x: x.rolling(5, min_periods=5).sum()))
    r = df_w_s.groupby(['y','x']).agg(max_rolling5_tp=('tp_r5','max')).reset_index()
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

# ── 3. Build raster grid ──────────────────────────────────────────────
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
    flow_dir   = np.zeros((H, W), dtype=np.float32)
    feat_vals  = df[feature_cols].fillna(0).values.astype(np.float32)
    labels     = df[target_col].values
    fdir       = df['flow_dir'].fillna(0).values
    yr_idx     = df['yr'].map(yr_to_i).values
    xr_idx     = df['xr'].map(xr_to_j).values
    for k in range(len(df)):
        i, j = yr_idx[k], xr_idx[k]
        feat_grid[i, j, :]  = feat_vals[k]
        flow_dir[i, j]       = fdir[k]
        lbl = labels[k]
        if not np.isnan(lbl) and lbl in [1,2,3,4]:
            label_grid[i, j] = int(lbl) - 1
    return feat_grid, label_grid, flow_dir, yr_vals, xr_vals

print("\nBuilding Severn grid...")
df_s_clean = df_s.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
grid_s, labels_s, fdir_s, yr_s, xr_s = df_to_grid(df_s_clean, FEATURE_COLS, TARGET_COL)

print("Building Northumbria grid...")
df_n_clean = df_n.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
grid_n, labels_n, fdir_n, yr_n, xr_n = df_to_grid(df_n_clean, FEATURE_COLS, TARGET_COL)

# ── 4. Build hydrological graph over basins ───────────────────────────
def build_basin_graph(feat_grid, label_grid, flow_dir_grid, basin_size=50):
    """
    Partition raster into basin blocks.
    Each basin = one graph node.
    Node features = mean of pixel features within basin.
    Node label = majority vote of pixel labels within basin.
    
    Edges represent hydrological connectivity:
    1. Spatial adjacency (8-neighbor between basins)
    2. Flow direction edges — dominant flow direction of basin
       determines downstream basin connection
    
    Returns torch_geometric Data object.
    """
    H, W, C = feat_grid.shape
    bs      = basin_size
    
    # Basin grid dimensions
    n_rows = H // bs
    n_cols = W // bs
    n_nodes = n_rows * n_cols
    
    print(f"  Basin grid: {n_rows} × {n_cols} = {n_nodes} nodes")
    print(f"  Each node covers {bs*20}m × {bs*20}m = {bs*20/1000:.1f}km²")
    
    # Node features and labels
    node_feats  = np.zeros((n_nodes, C), dtype=np.float32)
    node_labels = np.full(n_nodes, -1, dtype=np.int8)
    node_valid  = np.zeros(n_nodes, dtype=bool)
    node_flow   = np.zeros(n_nodes, dtype=np.float32)
    node_coords = np.zeros((n_nodes, 2), dtype=np.float32)  # (row, col) center
    
    for bi in range(n_rows):
        for bj in range(n_cols):
            node_id = bi * n_cols + bj
            r0, r1  = bi*bs, min((bi+1)*bs, H)
            c0, c1  = bj*bs, min((bj+1)*bs, W)
            
            patch_feats  = feat_grid[r0:r1, c0:c1, :]    # (bs, bs, C)
            patch_labels = label_grid[r0:r1, c0:c1]       # (bs, bs)
            patch_flow   = flow_dir_grid[r0:r1, c0:c1]    # (bs, bs)
            
            # Node feature = mean of all pixels in basin
            node_feats[node_id]  = patch_feats.reshape(-1, C).mean(axis=0)
            node_coords[node_id] = [(r0+r1)/2, (c0+c1)/2]
            
            # Mean flow direction of basin
            node_flow[node_id] = patch_flow.mean()
            
            # Node label = majority vote of valid pixels
            valid_pixels = patch_labels[patch_labels >= 0]
            if len(valid_pixels) > 0:
                counts = np.bincount(valid_pixels, minlength=4)
                node_labels[node_id] = counts.argmax()
                node_valid[node_id]  = True
    
    # Build edges
    edge_src, edge_dst = [], []
    edge_attr = []  # edge type: 0=spatial, 1=flow
    
    # D-infinity flow direction mapping
    # Degrees → (delta_row, delta_col)
    def flow_to_neighbor(angle_deg):
        """Map flow direction angle to downstream neighbor offset."""
        angle = angle_deg % 360
        if   angle < 22.5  or angle >= 337.5: return (0,  1)   # E
        elif angle < 67.5:                     return (-1, 1)   # NE
        elif angle < 112.5:                    return (-1, 0)   # N
        elif angle < 157.5:                    return (-1,-1)   # NW
        elif angle < 202.5:                    return (0, -1)   # W
        elif angle < 247.5:                    return (1, -1)   # SW
        elif angle < 292.5:                    return (1,  0)   # S
        else:                                  return (1,  1)   # SE
    
    for bi in range(n_rows):
        for bj in range(n_cols):
            src = bi * n_cols + bj
            if not node_valid[src]:
                continue
            
            # 1. Spatial adjacency edges (8-neighbor)
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
                            edge_attr.append(0)  # spatial
            
            # 2. Hydrological flow edge — basin drains downstream
            dr, dc = flow_to_neighbor(node_flow[src])
            ni, nj = bi+dr, bj+dc
            if 0 <= ni < n_rows and 0 <= nj < n_cols:
                dst = ni * n_cols + nj
                if node_valid[dst] and dst != src:
                    edge_src.append(src)
                    edge_dst.append(dst)
                    edge_attr.append(1)  # hydrological flow
    
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
    x          = torch.tensor(node_feats,  dtype=torch.float)
    y          = torch.tensor(node_labels, dtype=torch.long)
    valid_mask = torch.tensor(node_valid,  dtype=torch.bool)
    coords     = torch.tensor(node_coords, dtype=torch.float)
    
    print(f"  Valid nodes: {node_valid.sum():,}")
    print(f"  Edges: {len(edge_src):,} "
          f"(spatial: {sum(1 for e in edge_attr.numpy() if e==0):,} | "
          f"flow: {sum(1 for e in edge_attr.numpy() if e==1):,})")
    
    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        y=y, valid_mask=valid_mask, coords=coords,
        n_rows=n_rows, n_cols=n_cols
    )

print("\nBuilding Severn basin graph...")
graph_s = build_basin_graph(grid_s, labels_s, fdir_s, BASIN_SIZE)

print("\nBuilding Northumbria basin graph...")
graph_n = build_basin_graph(grid_n, labels_n, fdir_n, BASIN_SIZE)

# ── 5. CNN patch extractor ─────────────────────────────────────────────
class LocalCNNExtractor(nn.Module):
    """
    Extracts local spatial features from raster patches.
    Called once per node — processes BASIN_SIZE×BASIN_SIZE patch.
    Output: feature vector per node.
    """
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            # Input: (B, C, bs, bs)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # bs/2
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # bs/4
            nn.Dropout2d(0.1),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.proj   = nn.Linear(256, out_dim)
        self.relu   = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.encoder(x)
        x = self.gap(x).flatten(1)
        return self.relu(self.proj(x))

# ── 6. GNN message passing module ─────────────────────────────────────
class HydroGNN(nn.Module):
    """
    Graph Attention Network over basin nodes.
    Propagates information along both spatial adjacency 
    and hydrological flow edges.
    
    Uses edge attributes to distinguish spatial vs flow messages.
    """
    def __init__(self, node_dim, hidden_dim=128, n_classes=4, n_layers=3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )
        
        # GAT layers with edge features
        self.gat_layers = nn.ModuleList([
            GATConv(
                hidden_dim, hidden_dim // 4,
                heads=4,
                edge_dim=1,             # edge type (spatial=0, flow=1)
                concat=True,
                dropout=0.2
            )
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x, edge_index, edge_attr):
        x = self.input_proj(x)
        
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            residual = x
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = ln(x + residual)   # residual connection
        
        return self.classifier(x)

# ── 7. Full hybrid model ───────────────────────────────────────────────
class HybridCNNGNN(nn.Module):
    """
    Hybrid architecture:
    1. CNN extracts local spatial features from each basin's raster patch
    2. GNN propagates information along hydrological graph
    3. Classifier predicts flood risk per basin node
    
    The CNN sees local terrain texture.
    The GNN sees how water flows between basins.
    Together they capture both local and catchment-scale flood dynamics.
    """
    def __init__(self, in_channels, cnn_dim=64, gnn_hidden=128, n_classes=4):
        super().__init__()
        self.cnn = LocalCNNExtractor(in_channels, out_dim=cnn_dim)
        
        # GNN input = CNN features + raw node features (basin-level aggregates)
        gnn_input_dim = cnn_dim + in_channels
        self.gnn = HydroGNN(gnn_input_dim, gnn_hidden, n_classes)
    
    def forward(self, patches, x_node, edge_index, edge_attr):
        """
        patches    : (N_nodes, C, bs, bs) — raster patches per basin
        x_node     : (N_nodes, C)         — aggregated node features
        edge_index : (2, E)               — graph edges
        edge_attr  : (E, 1)               — edge type
        """
        # CNN: local feature extraction per node
        cnn_feats = self.cnn(patches)              # (N, cnn_dim)
        
        # Concatenate CNN features with aggregated basin stats
        node_feats = torch.cat([cnn_feats, x_node], dim=1)  # (N, cnn_dim + C)
        
        # GNN: propagate along hydrological graph
        logits = self.gnn(node_feats, edge_index, edge_attr)  # (N, n_classes)
        
        return logits

# ── 8. Build patch tensors for CNN ────────────────────────────────────
def extract_basin_patches(feat_grid, n_rows, n_cols, basin_size):
    """
    Extract raster patches for each basin node.
    Returns tensor of shape (n_nodes, C, bs, bs).
    """
    H, W, C = feat_grid.shape
    bs      = basin_size
    patches = []
    
    for bi in range(n_rows):
        for bj in range(n_cols):
            r0 = bi * bs
            c0 = bj * bs
            r1 = min(r0 + bs, H)
            c1 = min(c0 + bs, W)
            
            patch = feat_grid[r0:r1, c0:c1, :]   # (bs, bs, C)
            
            # Pad to exact bs×bs if at boundary
            if patch.shape[0] < bs or patch.shape[1] < bs:
                pad = np.zeros((bs, bs, C), dtype=np.float32)
                pad[:patch.shape[0], :patch.shape[1], :] = patch
                patch = pad
            
            patches.append(patch.transpose(2, 0, 1))  # (C, bs, bs)
    
    return torch.tensor(np.stack(patches), dtype=torch.float)

print("\nExtracting basin patches for CNN...")
print("Severn patches:")
patches_s = extract_basin_patches(
    grid_s, graph_s.n_rows, graph_s.n_cols, BASIN_SIZE
)
print(f"  Shape: {patches_s.shape}")

print("Northumbria patches:")
patches_n = extract_basin_patches(
    grid_n, graph_n.n_rows, graph_n.n_cols, BASIN_SIZE
)
print(f"  Shape: {patches_n.shape}")

# ── 9. Train/val split — random node split ────────────────────────────
valid_node_ids = graph_s.valid_mask.nonzero(as_tuple=True)[0].numpy()
n_valid        = len(valid_node_ids)
rng            = np.random.default_rng(42)
shuffled       = rng.permutation(valid_node_ids)
n_val          = int(n_valid * 0.2)
val_nodes      = set(shuffled[:n_val].tolist())
train_nodes    = set(shuffled[n_val:].tolist())

train_mask = torch.zeros(graph_s.x.shape[0], dtype=torch.bool)
val_mask   = torch.zeros(graph_s.x.shape[0], dtype=torch.bool)
for nid in train_nodes:
    train_mask[nid] = True
for nid in val_nodes:
    val_mask[nid] = True

print(f"\nTrain nodes: {train_mask.sum().item():,}")
print(f"Val nodes:   {val_mask.sum().item():,}")
print(f"Test nodes (Northumbria): {graph_n.valid_mask.sum().item():,}")

# ── 10. Training loop ─────────────────────────────────────────────────
model = HybridCNNGNN(
    in_channels=N_CHANNELS,
    cnn_dim=64,
    gnn_hidden=128,
    n_classes=4
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,}")

# Class weights from train nodes
train_labels_list = graph_s.y[train_mask].numpy()
train_labels_list = train_labels_list[train_labels_list >= 0]
class_counts = Counter(train_labels_list.tolist())
total_c      = sum(class_counts.values())
cw = torch.tensor(
    [total_c / (4 * class_counts.get(c, 1)) for c in range(4)],
    dtype=torch.float32
).to(device)
print(f"Class weights: {cw.cpu().numpy().round(3)}")

criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
scaler    = torch.amp.GradScaler('cuda')

# Move graph data to device
graph_s_dev    = graph_s.to(device)
patches_s_dev  = patches_s.to(device)
graph_n_dev    = graph_n.to(device)
patches_n_dev  = patches_n.to(device)
train_mask_dev = train_mask.to(device)
val_mask_dev   = val_mask.to(device)

def run_forward(model, patches, graph, mask=None):
    """Single forward pass on full graph."""
    with torch.amp.autocast('cuda'):
        logits = model(
            patches,
            graph.x,
            graph.edge_index,
            graph.edge_attr
        )
    if mask is not None:
        return logits[mask], graph.y[mask]
    return logits, graph.y

print(f"\nTraining for {N_EPOCHS} epochs...")
print(f"{'Epoch':>5} {'TrLoss':>8} {'TrAcc':>7} {'VlLoss':>8} {'VlAcc':>7} {'Time':>7}")
print("-" * 50)

best_val_acc = 0
best_state   = None

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    
    # ── Train ──────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    
    logits_tr, labels_tr = run_forward(
        model, patches_s_dev, graph_s_dev, train_mask_dev
    )
    # Filter valid labels
    valid = labels_tr >= 0
    loss  = criterion(logits_tr[valid], labels_tr[valid])
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    
    tr_acc = (logits_tr[valid].argmax(1) == labels_tr[valid]).float().mean().item()
    tr_loss = loss.item()
    
    # ── Validate ───────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits_vl, labels_vl = run_forward(
            model, patches_s_dev, graph_s_dev, val_mask_dev
        )
        valid_vl = labels_vl >= 0
        vl_loss  = criterion(logits_vl[valid_vl], labels_vl[valid_vl]).item()
        vl_acc   = (logits_vl[valid_vl].argmax(1) == labels_vl[valid_vl]).float().mean().item()
    
    elapsed = time.time() - t0
    marker  = ' ◄' if vl_acc > best_val_acc else ''
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    print(f"{epoch:>5} {tr_loss:>8.4f} {tr_acc:>7.3f} "
          f"{vl_loss:>8.4f} {vl_acc:>7.3f} {elapsed:>6.1f}s{marker}")

# ── 11. Final evaluation ──────────────────────────────────────────────
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
model.eval()

with torch.no_grad():
    # Severn val
    logits_vl, labels_vl = run_forward(
        model, patches_s_dev, graph_s_dev, val_mask_dev
    )
    valid_vl  = labels_vl >= 0
    vl_preds  = logits_vl[valid_vl].argmax(1).cpu().numpy()
    vl_true   = labels_vl[valid_vl].cpu().numpy()
    
    # Northumbria test
    logits_ts, labels_ts = run_forward(
        model, patches_n_dev, graph_n_dev
    )
    valid_ts  = (graph_n_dev.valid_mask) & (labels_ts >= 0)
    ts_preds  = logits_ts[valid_ts].argmax(1).cpu().numpy()
    ts_true   = labels_ts[valid_ts].cpu().numpy()

print("\n── Hybrid CNN-GNN: Severn val (seen) ──")
print(f"Accuracy: {accuracy_score(vl_true, vl_preds):.3f}")
print(classification_report(vl_true, vl_preds,
      target_names=['Very Low','Low','Medium','High']))

print("── Hybrid CNN-GNN: Northumbria test (unseen) ──")
print(f"Accuracy: {accuracy_score(ts_true, ts_preds):.3f}")
print(classification_report(ts_true, ts_preds,
      target_names=['Very Low','Low','Medium','High']))

torch.save(best_state, '/workspace/Flood-Risk/flood_hybrid_cnn_gnn.pt')
print("\nModel saved.")