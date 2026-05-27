import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import classification_report, accuracy_score
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
BATCH_SIZE = 4096       # larger batch = more stable gradients
N_EPOCHS   = 40
LR         = 3e-4       # lower LR = less overfitting
TARGET_COL = 'risk_0_2m'

FEATURE_COLS = [
    'dtm_zscore', 'log_flow_acc', 'imd', 'waw',
    'is_waterway', 'clc_type_clean',
    'tp_p99_zscore', 'max_rolling5_tp_zscore',
    'sro_p95_zscore', 'swvl1_min_zscore'
]
N_CHANNELS = len(FEATURE_COLS)

# ── 2. Load + feature engineering (same as before) ───────────────────
print("\nLoading datasets...")
ds_terrain_severn      = xr.open_dataset(DATA_DIR + 'Copy of Copy of flood_risk_terrain_severn.nc',      engine='netcdf4')
ds_terrain_northumbria = xr.open_dataset(DATA_DIR + 'Copy of Copy of flood_risk_terrain_northumbria.nc', engine='netcdf4')
ds_era5_severn         = xr.open_dataset(DATA_DIR + 'Copy of Copy of era5_land_severn.nc',               engine='netcdf4')
ds_era5_northumbria    = xr.open_dataset(DATA_DIR + 'Copy of Copy of era5_land_northumbria.nc',           engine='netcdf4')

risk_cols = ['risk_0_2m', 'risk_0_3m', 'risk_0_6m', 'risk_0_9m', 'risk_1_2m']

def filter_risk_pixels(df):
    mask = df[risk_cols].notna().any(axis=1) & (df[risk_cols] != 0).any(axis=1)
    return df[mask].copy()

def engineer_terrain_features(df):
    df = df.copy()
    df['waw']            = df['waw'].where(df['waw'] <= 4, np.nan).fillna(0)
    df['imd']            = df['imd'].where(df['imd'] <= 100, np.nan).fillna(0)
    df['dtm_m']          = df['dtm'] / 10
    df['dtm_zscore']     = (df['dtm_m'] - df['dtm_m'].mean()) / df['dtm_m'].std()
    df['log_flow_acc']   = np.log1p(df['flow_acc'])
    df['is_waterway']    = df['rciw'].notna().astype(int)
    df['clc_type_clean'] = df['clc_type'].fillna(df['clc_type'].median())
    return df

def engineer_weather_features(ds_era5):
    df_w = ds_era5[['tp', 'sro', 'swvl1_mean', 'swvl1_max']].to_dataframe().reset_index()
    df_w = df_w.dropna(subset=['tp'])
    grouped = df_w.groupby(['y', 'x'])
    weather = grouped.agg(
        tp_p99    = ('tp',        lambda x: np.percentile(x, 99)),
        sro_p95   = ('sro',       lambda x: np.percentile(x, 95)),
        swvl1_min = ('swvl1_mean','min'),
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

def add_relative_weather(df, weather_cols):
    df = df.copy()
    for col in weather_cols:
        mean = df[col].mean()
        std  = df[col].std()
        df[f'{col}_zscore'] = (df[col] - mean) / (std + 1e-8)
    return df

def merge_weather_to_terrain(df_terrain, df_weather):
    from scipy.spatial import cKDTree
    tree = cKDTree(df_weather[['y', 'x']].values)
    _, indices = tree.query(df_terrain[['y', 'x']].values, k=1)
    weather_cols = [c for c in df_weather.columns if c not in ['y', 'x']]
    matched = df_weather.iloc[indices][weather_cols].reset_index(drop=True)
    return pd.concat([df_terrain.reset_index(drop=True), matched], axis=1)

# Severn
print("Processing Severn...")
df_s = ds_terrain_severn.to_dataframe().reset_index()
df_s = filter_risk_pixels(df_s)
df_s = df_s[df_s['risk_0_2m'].isin([1.,2.,3.,4.])].copy()
df_s = engineer_terrain_features(df_s)
weather_s = engineer_weather_features(ds_era5_severn)
df_s = merge_weather_to_terrain(df_s, weather_s)
df_s = add_relative_weather(df_s, ['tp_p99','sro_p95','swvl1_min','max_rolling5_tp'])
print(f"Severn: {len(df_s):,} pixels")

# Northumbria
print("Processing Northumbria...")
df_n = ds_terrain_northumbria.to_dataframe().reset_index()
df_n = filter_risk_pixels(df_n)
df_n = engineer_terrain_features(df_n)
weather_n = engineer_weather_features(ds_era5_northumbria)
df_n = merge_weather_to_terrain(df_n, weather_n)
df_n = add_relative_weather(df_n, ['tp_p99','sro_p95','swvl1_min','max_rolling5_tp'])
print(f"Northumbria: {len(df_n):,} pixels")

# ── 3. Build raster grid ──────────────────────────────────────────────
def df_to_grid(df, feature_cols, target_col, resolution=20):
    df = df.copy()
    df['yr'] = (df['y'] / resolution).round().astype(int)
    df['xr'] = (df['x'] / resolution).round().astype(int)
    df = df.drop_duplicates(subset=['yr','xr'])
    yr_vals  = np.sort(df['yr'].unique())[::-1]
    xr_vals  = np.sort(df['xr'].unique())
    yr_to_i  = {v: i for i, v in enumerate(yr_vals)}
    xr_to_j  = {v: j for j, v in enumerate(xr_vals)}
    H, W, C  = len(yr_vals), len(xr_vals), len(feature_cols)
    print(f"  Grid: {H} × {W} = {H*W:,} cells | {C} channels")
    feat_grid  = np.zeros((H, W, C), dtype=np.float32)
    label_grid = np.full((H, W), -1, dtype=np.int8)
    feat_vals  = df[feature_cols].fillna(0).values.astype(np.float32)
    labels     = df[target_col].values
    yr_idx     = df['yr'].map(yr_to_i).values
    xr_idx     = df['xr'].map(xr_to_j).values
    for k in range(len(df)):
        i, j = yr_idx[k], xr_idx[k]
        feat_grid[i, j, :] = feat_vals[k]
        lbl = labels[k]
        if not np.isnan(lbl) and lbl in [1,2,3,4]:
            label_grid[i, j] = int(lbl) - 1
    return feat_grid, label_grid

print("\nBuilding Severn grid...")
df_s_clean = df_s.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
grid_s, labels_s = df_to_grid(df_s_clean, FEATURE_COLS, TARGET_COL)

print("Building Northumbria grid...")
df_n_clean = df_n.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
grid_n, labels_n = df_to_grid(df_n_clean, FEATURE_COLS, TARGET_COL)

# ── 4. Patch Dataset — KEY FIX: random spatial split ─────────────────
class FloodPatchDataset(Dataset):
    def __init__(self, feat_grid, label_grid, patch_size=11,
                 augment=False, val_fraction=0.0, is_val=False, seed=42):
        self.feat  = feat_grid
        self.label = label_grid
        self.P     = patch_size
        self.half  = patch_size // 2
        self.aug   = augment
        H, W       = label_grid.shape

        ys, xs = np.where(
            (label_grid >= 0) &
            (np.arange(H)[:, None] >= self.half) &
            (np.arange(H)[:, None] <  H - self.half) &
            (np.arange(W)[None, :] >= self.half) &
            (np.arange(W)[None, :] <  W - self.half)
        )
        all_positions = list(zip(ys.tolist(), xs.tolist()))

        # Random split instead of spatial split
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
        print(f"  Patches: {len(self.positions):,}")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        i, j  = self.positions[idx]
        half  = self.half
        patch = self.feat[i-half:i+half+1, j-half:j+half+1, :]
        patch = torch.from_numpy(patch.transpose(2, 0, 1))
        label = int(self.label[i, j])
        if self.aug:
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[2])
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[1])
            if torch.rand(1) > 0.5:
                patch = torch.rot90(patch, k=1, dims=[1, 2])
        return patch, label

# ── 5. CNN — deeper with residual connections ─────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class FloodCNN(nn.Module):
    def __init__(self, in_channels, n_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )
        self.layer2 = nn.Sequential(
            ResBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.gap(x)
        return self.classifier(x)

# ── 6. Train / eval ───────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for patches, labels in loader:
        patches = patches.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(patches)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        correct    += (logits.detach().argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(patches)
                loss   = criterion(logits, labels)
            preds       = logits.argmax(1)
            total_loss += loss.item() * len(labels)
            correct    += (preds == labels).sum().item()
            total      += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels

# ── 7. Build datasets ─────────────────────────────────────────────────
print("\nBuilding datasets...")
print("Train dataset:")
train_ds = FloodPatchDataset(grid_s, labels_s, PATCH_SIZE,
                             augment=True, val_fraction=0.2,
                             is_val=False)
print("Val dataset:")
val_ds   = FloodPatchDataset(grid_s, labels_s, PATCH_SIZE,
                             augment=False, val_fraction=0.2,
                             is_val=True)
print("Northumbria test dataset:")
test_ds  = FloodPatchDataset(grid_n, labels_n, PATCH_SIZE,
                             augment=False)

# Weighted sampler — oversample minority classes in training
train_labels = [train_ds.label[i][j] for i, j in train_ds.positions]
class_counts = Counter(train_labels)
total_samples = len(train_labels)
sample_weights = [total_samples / (4 * class_counts[train_ds.label[i][j]])
                  for i, j in train_ds.positions]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler,
                          num_workers=8, pin_memory=True, prefetch_factor=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2,
                          shuffle=False,
                          num_workers=8, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2,
                          shuffle=False,
                          num_workers=8, pin_memory=True)

# ── 8. Model, optimizer, scheduler ───────────────────────────────────
model     = FloodCNN(N_CHANNELS, n_classes=4).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing helps generalization
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=N_EPOCHS,
    pct_start=0.1
)
scaler = torch.amp.GradScaler('cuda')  # mixed precision for speed

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,}")

# ── 9. Training loop ─────────────────────────────────────────────────
print(f"\nTraining for {N_EPOCHS} epochs on {device}...")
print(f"{'Epoch':>5} {'TrLoss':>8} {'TrAcc':>7} {'VlLoss':>8} {'VlAcc':>7} {'Time':>7}")
print("-" * 50)

best_val_acc = 0
best_state   = None

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(
        model, train_loader, optimizer, criterion, device, scaler)
    vl_loss, vl_acc, vl_preds, vl_true = eval_epoch(
        model, val_loader, criterion, device)
    scheduler.step()
    elapsed = time.time() - t0

    marker = ' ◄' if vl_acc > best_val_acc else ''
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"{epoch:>5} {tr_loss:>8.4f} {tr_acc:>7.3f} "
          f"{vl_loss:>8.4f} {vl_acc:>7.3f} {elapsed:>6.1f}s{marker}")

# ── 10. Final evaluation ──────────────────────────────────────────────
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

_, _, vl_preds, vl_true = eval_epoch(model, val_loader,  criterion, device)
_, _, ts_preds, ts_true = eval_epoch(model, test_loader, criterion, device)

print("\n── CNN v2: Severn val (seen) ──")
print(f"Accuracy: {accuracy_score(vl_true, vl_preds):.3f}")
print(classification_report(vl_true, vl_preds,
      target_names=['Very Low', 'Low', 'Medium', 'High']))

print("── CNN v2: Northumbria test (unseen) ──")
print(f"Accuracy: {accuracy_score(ts_true, ts_preds):.3f}")
print(classification_report(ts_true, ts_preds,
      target_names=['Very Low', 'Low', 'Medium', 'High']))

torch.save(best_state, '/workspace/Flood-Risk/flood_cnn_v2_best.pt')
print("\nModel saved.")