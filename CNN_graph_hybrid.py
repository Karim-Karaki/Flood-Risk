import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ── 0. Device ─────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── 1. Config ─────────────────────────────────────────────────────────
DATA_DIR = '/workspace/Data-Flood/'
WINDOW_SIZE = 9          # graph window size in pixels (odd number)
HALF = WINDOW_SIZE // 2
BATCH_SIZE = 256
N_EPOCHS = 40
LR = 3e-4
TARGET_COL = 'risk_0_2m'

FEATURE_COLS = [
    'dtm_zscore', 'log_flow_acc', 'imd', 'waw',
    'is_waterway', 'clc_type_clean',
    'tp_p99_zscore', 'max_rolling5_tp_zscore',
    'sro_p95_zscore', 'swvl1_min_zscore'
]
N_CHANNELS = len(FEATURE_COLS)

RISK_COLS = ['risk_0_2m', 'risk_0_3m', 'risk_0_6m', 'risk_0_9m', 'risk_1_2m']


# ── 2. Load + feature engineering ─────────────────────────────────────
print("\nLoading datasets...")
ds_terrain_severn = xr.open_dataset(
    DATA_DIR + 'Copy of Copy of flood_risk_terrain_severn.nc',
    engine='netcdf4'
)
ds_terrain_northumbria = xr.open_dataset(
    DATA_DIR + 'Copy of Copy of flood_risk_terrain_northumbria.nc',
    engine='netcdf4'
)
ds_era5_severn = xr.open_dataset(
    DATA_DIR + 'Copy of Copy of era5_land_severn.nc',
    engine='netcdf4'
)
ds_era5_northumbria = xr.open_dataset(
    DATA_DIR + 'Copy of Copy of era5_land_northumbria.nc',
    engine='netcdf4'
)


def filter_risk_pixels(df: pd.DataFrame) -> pd.DataFrame:
    mask = df[RISK_COLS].notna().any(axis=1) & (df[RISK_COLS] != 0).any(axis=1)
    return df[mask].copy()


def engineer_terrain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['waw'] = df['waw'].where(df['waw'] <= 4, np.nan).fillna(0)
    df['imd'] = df['imd'].where(df['imd'] <= 100, np.nan).fillna(0)
    df['dtm_m'] = df['dtm'] / 10
    df['dtm_zscore'] = (df['dtm_m'] - df['dtm_m'].mean()) / (df['dtm_m'].std() + 1e-8)
    df['log_flow_acc'] = np.log1p(df['flow_acc'])
    df['is_waterway'] = df['rciw'].notna().astype(int)
    df['clc_type_clean'] = df['clc_type'].fillna(df['clc_type'].median())
    return df


def engineer_weather_features(ds_era5: xr.Dataset) -> pd.DataFrame:
    df_w = ds_era5[['tp', 'sro', 'swvl1_mean', 'swvl1_max']].to_dataframe().reset_index()
    df_w = df_w.dropna(subset=['tp'])
    grouped = df_w.groupby(['y', 'x'])
    weather = grouped.agg(
        tp_p99=('tp', lambda x: np.percentile(x, 99)),
        sro_p95=('sro', lambda x: np.percentile(x, 95)),
        swvl1_min=('swvl1_mean', 'min'),
    ).reset_index()

    df_w_sorted = df_w.sort_values(['y', 'x', 'valid_time'])
    df_w_sorted['tp_rolling5'] = (
        df_w_sorted.groupby(['y', 'x'])['tp']
        .transform(lambda x: x.rolling(5, min_periods=5).sum())
    )
    rolling = df_w_sorted.groupby(['y', 'x']).agg(
        max_rolling5_tp=('tp_rolling5', 'max')
    ).reset_index()

    return weather.merge(rolling, on=['y', 'x'])


def add_relative_weather(df: pd.DataFrame, weather_cols):
    df = df.copy()
    for col in weather_cols:
        mean = df[col].mean()
        std = df[col].std() + 1e-8
        df[f'{col}_zscore'] = (df[col] - mean) / std
    return df


def merge_weather_to_terrain(df_terrain: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    from scipy.spatial import cKDTree
    tree = cKDTree(df_weather[['y', 'x']].values)
    _, indices = tree.query(df_terrain[['y', 'x']].values, k=1)
    weather_cols = [c for c in df_weather.columns if c not in ['y', 'x']]
    matched = df_weather.iloc[indices][weather_cols].reset_index(drop=True)
    return pd.concat([df_terrain.reset_index(drop=True), matched], axis=1)


print("Processing Severn...")
df_s = ds_terrain_severn.to_dataframe().reset_index()
df_s = filter_risk_pixels(df_s)
df_s = df_s[df_s['risk_0_2m'].isin([1., 2., 3., 4.])].copy()
df_s = engineer_terrain_features(df_s)
weather_s = engineer_weather_features(ds_era5_severn)
df_s = merge_weather_to_terrain(df_s, weather_s)
df_s = add_relative_weather(df_s, ['tp_p99', 'sro_p95', 'swvl1_min', 'max_rolling5_tp'])
print(f"Severn: {len(df_s):,} pixels")

print("Processing Northumbria...")
df_n = ds_terrain_northumbria.to_dataframe().reset_index()
df_n = filter_risk_pixels(df_n)
df_n = engineer_terrain_features(df_n)
weather_n = engineer_weather_features(ds_era5_northumbria)
df_n = merge_weather_to_terrain(df_n, weather_n)
df_n = add_relative_weather(df_n, ['tp_p99', 'sro_p95', 'swvl1_min', 'max_rolling5_tp'])
print(f"Northumbria: {len(df_n):,} pixels")


# ── 3. Build raster grid ──────────────────────────────────────────────
def df_to_grid(df: pd.DataFrame, feature_cols, target_col, resolution=20):
    df = df.copy()
    df['yr'] = (df['y'] / resolution).round().astype(int)
    df['xr'] = (df['x'] / resolution).round().astype(int)
    df = df.drop_duplicates(subset=['yr', 'xr'])

    yr_vals = np.sort(df['yr'].unique())[::-1]
    xr_vals = np.sort(df['xr'].unique())
    yr_to_i = {v: i for i, v in enumerate(yr_vals)}
    xr_to_j = {v: j for j, v in enumerate(xr_vals)}

    H, W, C = len(yr_vals), len(xr_vals), len(feature_cols)
    print(f"  Grid: {H} x {W} = {H * W:,} cells | {C} channels")

    feat_grid = np.zeros((H, W, C), dtype=np.float32)
    label_grid = np.full((H, W), -1, dtype=np.int8)

    feat_vals = df[feature_cols].fillna(0).values.astype(np.float32)
    labels = df[target_col].values
    yr_idx = df['yr'].map(yr_to_i).values
    xr_idx = df['xr'].map(xr_to_j).values

    for k in range(len(df)):
        i, j = yr_idx[k], xr_idx[k]
        feat_grid[i, j, :] = feat_vals[k]
        lbl = labels[k]
        if not np.isnan(lbl) and lbl in [1, 2, 3, 4]:
            label_grid[i, j] = int(lbl) - 1

    return feat_grid, label_grid


print("\nBuilding Severn grid...")
df_s_clean = df_s.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
grid_s, labels_s = df_to_grid(df_s_clean, FEATURE_COLS, TARGET_COL)

print("Building Northumbria grid...")
df_n_clean = df_n.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
grid_n, labels_n = df_to_grid(df_n_clean, FEATURE_COLS, TARGET_COL)


# ── 4. Graph window dataset ───────────────────────────────────────────
def build_grid_edge_index(window_size: int, eight_neigh: bool = True) -> torch.Tensor:
    idx = np.arange(window_size * window_size).reshape(window_size, window_size)
    if eight_neigh:
        offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    edges = []
    for r in range(window_size):
        for c in range(window_size):
            src = idx[r, c]
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < window_size and 0 <= cc < window_size:
                    dst = idx[rr, cc]
                    edges.append((src, dst))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


EDGE_INDEX_TEMPLATE = build_grid_edge_index(WINDOW_SIZE, eight_neigh=True)
NUM_NODES = WINDOW_SIZE * WINDOW_SIZE
CENTER_NODE = NUM_NODES // 2


class RasterGraphWindowDataset(Dataset):
    def __init__(self, feat_grid, label_grid, window_size=9, augment=False,
                 val_fraction=0.0, is_val=False, seed=42):
        self.feat = feat_grid
        self.label = label_grid
        self.W = window_size
        self.half = window_size // 2
        self.aug = augment

        H, W = label_grid.shape
        ys, xs = np.where(
            (label_grid >= 0) &
            (np.arange(H)[:, None] >= self.half) &
            (np.arange(H)[:, None] < H - self.half) &
            (np.arange(W)[None, :] >= self.half) &
            (np.arange(W)[None, :] < W - self.half)
        )
        all_positions = list(zip(ys.tolist(), xs.tolist()))

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(all_positions))
        n_val = int(len(all_positions) * val_fraction)

        if val_fraction > 0:
            if is_val:
                selected = [all_positions[i] for i in idx[:n_val]]
            else:
                selected = [all_positions[i] for i in idx[n_val:]]
        else:
            selected = all_positions

        self.positions = selected
        print(f"  Samples: {len(self.positions):,}")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        i, j = self.positions[idx]
        half = self.half
        window = self.feat[i-half:i+half+1, j-half:j+half+1, :].astype(np.float32)
        x = torch.from_numpy(window.reshape(-1, window.shape[-1]))  # [nodes, channels]
        y = int(self.label[i, j])

        if self.aug:
            # flip the node grid before flattening
            if torch.rand(1) > 0.5:
                window = np.flip(window, axis=1).copy()
            if torch.rand(1) > 0.5:
                window = np.flip(window, axis=0).copy()
            if torch.rand(1) > 0.5:
                window = np.rot90(window, k=1, axes=(0, 1)).copy()
            x = torch.from_numpy(window.reshape(-1, window.shape[-1]))

        return x, EDGE_INDEX_TEMPLATE.clone(), CENTER_NODE, y


def collate_graphs(batch):
    xs, edge_indices, centers, ys = zip(*batch)
    x_list = []
    edge_list = []
    center_list = []
    batch_list = []
    offset = 0
    for g_idx, (x, edge_index, center, y) in enumerate(zip(xs, edge_indices, centers, ys)):
        n = x.size(0)
        x_list.append(x)
        edge_list.append(edge_index + offset)
        center_list.append(center + offset)
        batch_list.append(torch.full((n,), g_idx, dtype=torch.long))
        offset += n
    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_list, dim=1)
    center_idx = torch.tensor(center_list, dtype=torch.long)
    batch_vec = torch.cat(batch_list, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return x, edge_index, center_idx, batch_vec, y


# ── 5. Graph model ────────────────────────────────────────────────────
def global_mean_pool(x, batch):
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    out = torch.zeros((num_graphs, x.size(1)), device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    count = torch.bincount(batch, minlength=num_graphs).to(x.dtype).unsqueeze(-1)
    return out / count.clamp_min(1.0)


class GraphBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim)
        self.lin_neigh = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=x.device, dtype=x.dtype))
        agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
        out = self.lin_self(x) + self.lin_neigh(agg)
        out = self.drop(self.act(self.norm(out)))
        return x + out


class FloodGraphNet(nn.Module):
    def __init__(self, in_channels, n_classes=4, hidden_dim=96, depth=4, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([GraphBlock(hidden_dim, dropout=dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 192),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(192, n_classes),
        )

    def forward(self, x, edge_index, center_idx, batch):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, edge_index)

        graph_mean = global_mean_pool(x, batch)
        center_repr = x[center_idx]
        z = torch.cat([center_repr, graph_mean], dim=1)
        return self.head(z)


# ── 6. Train / eval ───────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, edge_index, center_idx, batch_vec, labels in loader:
        x = x.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        center_idx = center_idx.to(device, non_blocking=True)
        batch_vec = batch_vec.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(x, edge_index, center_idx, batch_vec)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        correct += (logits.detach().argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, edge_index, center_idx, batch_vec, labels in loader:
        x = x.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        center_idx = center_idx.to(device, non_blocking=True)
        batch_vec = batch_vec.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(x, edge_index, center_idx, batch_vec)
            loss = criterion(logits, labels)

        preds = logits.argmax(1)
        total_loss += loss.item() * len(labels)
        correct += (preds == labels).sum().item()
        total += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ── 7. Build datasets ─────────────────────────────────────────────────
print("\nBuilding datasets...")
print("Train dataset:")
train_ds = RasterGraphWindowDataset(
    grid_s, labels_s, WINDOW_SIZE,
    augment=True, val_fraction=0.2, is_val=False
)
print("Val dataset:")
val_ds = RasterGraphWindowDataset(
    grid_s, labels_s, WINDOW_SIZE,
    augment=False, val_fraction=0.2, is_val=True
)
print("Northumbria test dataset:")
test_ds = RasterGraphWindowDataset(
    grid_n, labels_n, WINDOW_SIZE,
    augment=False
)

train_labels = [train_ds.label[i][j] for i, j in train_ds.positions]
class_counts = Counter(train_labels)
total_samples = len(train_labels)
sample_weights = [total_samples / (4 * class_counts[train_ds.label[i][j]]) for i, j in train_ds.positions]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_graphs,
    persistent_workers=True
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_graphs,
    persistent_workers=True
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_graphs,
    persistent_workers=True
)


# ── 8. Model, optimizer, scheduler ───────────────────────────────────
model = FloodGraphNet(N_CHANNELS, n_classes=4, hidden_dim=96, depth=4, dropout=0.15).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=N_EPOCHS,
    pct_start=0.1
)
scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,}")


# ── 9. Training loop ─────────────────────────────────────────────────
print(f"\nTraining for {N_EPOCHS} epochs on {device}...")
print(f"{'Epoch':>5} {'TrLoss':>8} {'TrAcc':>7} {'VlLoss':>8} {'VlAcc':>7} {'Time':>7}")
print("-" * 50)

best_val_acc = 0.0
best_state = None

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
    vl_loss, vl_acc, vl_preds, vl_true = eval_epoch(model, val_loader, criterion, device)
    scheduler.step()
    elapsed = time.time() - t0

    marker = ' ◄' if vl_acc > best_val_acc else ''
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"{epoch:>5} {tr_loss:>8.4f} {tr_acc:>7.3f} {vl_loss:>8.4f} {vl_acc:>7.3f} {elapsed:>6.1f}s{marker}")


# ── 10. Final evaluation ──────────────────────────────────────────────
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

_, _, vl_preds, vl_true = eval_epoch(model, val_loader, criterion, device)
_, _, ts_preds, ts_true = eval_epoch(model, test_loader, criterion, device)

print("\n── Graph model: Severn val (seen) ──")
print(f"Accuracy: {accuracy_score(vl_true, vl_preds):.3f}")
print(classification_report(vl_true, vl_preds, target_names=['Very Low', 'Low', 'Medium', 'High']))

print("── Graph model: Northumbria test (unseen) ──")
print(f"Accuracy: {accuracy_score(ts_true, ts_preds):.3f}")
print(classification_report(ts_true, ts_preds, target_names=['Very Low', 'Low', 'Medium', 'High']))

out_path = '/workspace/Flood-Risk/flood_graph_hybrid_best.pt'
torch.save(best_state, out_path)
print(f"\nModel saved to {out_path}")
