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
BATCH_SIZE = 4096
N_EPOCHS   = 50
LR         = 3e-4

# All 5 depth levels as multi-output targets
DEPTH_COLS = ['risk_0_2m', 'risk_0_3m', 'risk_0_6m', 'risk_0_9m', 'risk_1_2m']
N_DEPTHS   = len(DEPTH_COLS)

# Map ordinal classes to exceedance probabilities
# Based on class definitions from the brief:
# Class 1 (Very Low) : <0.1% annual chance  → 0.001
# Class 2 (Low)      : 0.1-1% annual chance → 0.005
# Class 3 (Medium)   : 1-3.3% annual chance → 0.02
# Class 4 (High)     : >3.3% annual chance  → 0.05
# NaN / 0            : no risk              → 0.0
CLASS_TO_PROB = {
    0: 0.0,
    1: 0.001,   # Very Low  < 0.1% annual
    2: 0.01,    # Low       0.1-1%
    3: 0.033,   # Medium    1-3.3%
    4: 0.10,    # High      > 3.3%
}

FEATURE_COLS = [
    'dtm_zscore', 'log_flow_acc', 'imd', 'waw',
    'is_waterway', 'clc_type_clean',
    'tp_p99_zscore', 'max_rolling5_tp_zscore',
    'sro_p95_zscore', 'swvl1_min_zscore'
]
N_CHANNELS = len(FEATURE_COLS)

# ── 2. Load data ──────────────────────────────────────────────────────
print("\nLoading datasets...")
ds_terrain_severn      = xr.open_dataset(DATA_DIR + 'Copy of Copy of flood_risk_terrain_severn.nc',      engine='h5netcdf')
ds_terrain_northumbria = xr.open_dataset(DATA_DIR + 'Copy of Copy of flood_risk_terrain_northumbria.nc', engine='h5netcdf')
ds_era5_severn         = xr.open_dataset(DATA_DIR + 'Copy of Copy of era5_land_severn.nc',               engine='h5netcdf')
ds_era5_northumbria    = xr.open_dataset(DATA_DIR + 'Copy of Copy of era5_land_northumbria.nc',           engine='h5netcdf')
print("Loaded.")

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

# ── KEY: convert all depth labels to hazard curve probabilities ───────
def build_hazard_targets(df):
    """
    Convert 5 ordinal risk columns into exceedance probability vector.
    Output: (N, 5) array of probabilities, monotonically decreasing.
    Physical constraint: P(depth≥0.2m) >= P(depth≥0.3m) >= ... >= P(depth≥1.2m)
    """
    targets = np.zeros((len(df), N_DEPTHS), dtype=np.float32)
    for d, col in enumerate(DEPTH_COLS):
        vals = df[col].fillna(0).values
        probs = np.vectorize(CLASS_TO_PROB.get)(vals.astype(int))
        targets[:, d] = probs

    # Enforce monotone decreasing constraint
    # If deeper depth has higher prob than shallower (data artifact), clip it
    for d in range(1, N_DEPTHS):
        targets[:, d] = np.minimum(targets[:, d], targets[:, d-1])

    return targets
print("\nHazard target distribution check:")
classes_check = hazard_to_class(hazard_s_arr)
unique, counts = np.unique(classes_check, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:,} ({c/len(classes_check)*100:.1f}%)")

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

# ── 4. Build hazard curve targets ─────────────────────────────────────
print("\nBuilding hazard curve targets...")
hazard_s = build_hazard_targets(df_s)
hazard_n = build_hazard_targets(df_n)

print(f"Severn hazard shape: {hazard_s.shape}")
print(f"Sample hazard curves (first 5 pixels):")
print(hazard_s[:5])
print(f"Monotone check — violations: {(np.diff(hazard_s, axis=1) > 0).sum()}")

# ── 5. Build raster grid ──────────────────────────────────────────────
def df_to_grid_hazard(df, feature_cols, hazard_targets, resolution=20):
    df = df.copy()
    df['yr'] = (df['y'] / resolution).round().astype(int)
    df['xr'] = (df['x'] / resolution).round().astype(int)

    # Remove duplicates keeping first
    mask = ~df.duplicated(subset=['yr','xr'])
    df   = df[mask].copy()
    ht   = hazard_targets[mask]

    yr_vals = np.sort(df['yr'].unique())[::-1]
    xr_vals = np.sort(df['xr'].unique())
    yr_to_i = {v: i for i, v in enumerate(yr_vals)}
    xr_to_j = {v: j for j, v in enumerate(xr_vals)}

    H, W, C = len(yr_vals), len(xr_vals), len(feature_cols)
    print(f"  Grid: {H} × {W} = {H*W:,} cells")

    feat_grid   = np.zeros((H, W, C),        dtype=np.float32)
    hazard_grid = np.full((H, W, N_DEPTHS), -1.0, dtype=np.float32)

    feat_vals = df[feature_cols].fillna(0).values.astype(np.float32)
    yr_idx    = df['yr'].map(yr_to_i).values
    xr_idx    = df['xr'].map(xr_to_j).values

    for k in range(len(df)):
        i, j = yr_idx[k], xr_idx[k]
        feat_grid[i, j, :]   = feat_vals[k]
        hazard_grid[i, j, :] = ht[k]

    return feat_grid, hazard_grid

print("\nBuilding Severn grid...")
df_s_clean = df_s.dropna(subset=FEATURE_COLS).copy()
hazard_s_clean = hazard_s[df_s_clean.index - df_s_clean.index[0]
                          if df_s_clean.index[0] != 0
                          else df_s_clean.index]

# Simpler — reset index first
df_s = df_s.reset_index(drop=True)
hazard_s_arr = build_hazard_targets(df_s)
df_s_clean   = df_s.dropna(subset=FEATURE_COLS).copy()
hazard_s_clean = hazard_s_arr[df_s_clean.index.values]

grid_s, hazard_grid_s = df_to_grid_hazard(
    df_s_clean, FEATURE_COLS, hazard_s_clean
)

print("Building Northumbria grid...")
df_n = df_n.reset_index(drop=True)
hazard_n_arr = build_hazard_targets(df_n)
df_n_clean   = df_n.dropna(subset=FEATURE_COLS).copy()
hazard_n_clean = hazard_n_arr[df_n_clean.index.values]

grid_n, hazard_grid_n = df_to_grid_hazard(
    df_n_clean, FEATURE_COLS, hazard_n_clean
)

# ── 6. Patch Dataset ──────────────────────────────────────────────────
class HazardPatchDataset(torch.utils.data.Dataset):
    """
    Each sample: 11×11 patch of features → hazard curve (5 probabilities)
    Target is a vector not a class — this is the key difference.
    """
    def __init__(self, feat_grid, hazard_grid, patch_size=11,
                 augment=False, val_fraction=0.0, is_val=False, seed=42):
        self.feat   = feat_grid     # (H, W, C)
        self.hazard = hazard_grid   # (H, W, 5)
        self.P      = patch_size
        self.half   = patch_size // 2
        self.aug    = augment
        H, W        = hazard_grid.shape[:2]

        # Valid centers: has hazard data and not on edge
        valid = (hazard_grid[:, :, 0] >= 0)
        ys, xs = np.where(
            valid &
            (np.arange(H)[:, None] >= self.half) &
            (np.arange(H)[:, None] <  H - self.half) &
            (np.arange(W)[None, :] >= self.half) &
            (np.arange(W)[None, :] <  W - self.half)
        )
        all_pos = list(zip(ys.tolist(), xs.tolist()))

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(all_pos))
        n_val = int(len(all_pos) * val_fraction)

        if val_fraction > 0:
            selected = [all_pos[i] for i in (idx[:n_val] if is_val else idx[n_val:])]
        else:
            selected = all_pos

        self.positions = selected
        print(f"  Patches: {len(self.positions):,}")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        i, j   = self.positions[idx]
        half   = self.half
        patch  = self.feat[i-half:i+half+1, j-half:j+half+1, :]
        patch  = torch.from_numpy(patch.transpose(2, 0, 1))    # (C, P, P)
        hazard = torch.from_numpy(self.hazard[i, j, :].copy()) # (5,)

        if self.aug:
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[2])
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[1])

        return patch, hazard

# ── 7. Hazard curve CNN ───────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class HazardCNN(nn.Module):
    """
    CNN that predicts the full hazard curve per pixel.
    Output: 5 exceedance probabilities, forced monotone decreasing.

    Key innovation: instead of classifying into 4 discrete buckets,
    we predict the continuous hazard curve and enforce physical
    monotonicity constraint — deeper floods are always less likely.
    This matches how flood risk is actually measured (hazard curve)
    and provides much stronger regularization for cross-region transfer.
    """
    def __init__(self, in_channels, n_depths=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            ResBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Two-head output:
        # Head 1: base exceedance probability at 0.2m
        # Head 2: log-differences between consecutive depths
        #         (negative → ensures monotone decreasing after exp)
        self.head_base = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()          # P(depth≥0.2m) ∈ [0,1]
        )
        self.head_drops = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_depths - 1),
            nn.Softplus()         # positive drops → ensures monotone decrease
        )

    def forward(self, x):
        feat = self.gap(self.encoder(x))          # (B, 256, 1, 1)

        # Base probability at shallowest depth
        p_base = self.head_base(feat)             # (B, 1)

        # Positive drops at each subsequent depth
        drops  = self.head_drops(feat)            # (B, 4)

        # Build monotone hazard curve
        # P(d0) = p_base
        # P(d1) = P(d0) - drop0  (must be < P(d0))
        # P(d2) = P(d1) - drop1  etc.
        # Use cumulative sum of drops scaled to [0, p_base]
        # to ensure all values stay in [0,1] and monotone

        # Normalize drops to sum to at most p_base
        drop_sum   = drops.sum(dim=1, keepdim=True) + 1e-8
        drops_norm = drops / drop_sum * p_base      # scaled drops

        # Build curve via cumulative subtraction
        # curve[:, 0] = p_base
        # curve[:, k] = p_base - sum(drops_norm[:, :k])
        cum_drops = torch.cumsum(drops_norm, dim=1)  # (B, 4)
        p_rest    = p_base - cum_drops               # (B, 4)
        p_rest    = torch.clamp(p_rest, min=0.0)

        hazard_curve = torch.cat([p_base, p_rest], dim=1)  # (B, 5)

        return hazard_curve

# ── 8. Hazard-aware loss ──────────────────────────────────────────────
class HazardLoss(nn.Module):
    """
    Combined loss for hazard curve prediction:
    1. MSE on exceedance probabilities
    2. Monotonicity penalty (soft constraint)
    3. Ordinal classification loss — convert predictions back to
       risk classes and penalize misclassification
    """
    def __init__(self, lambda_mono=0.1, lambda_ord=1.0):
        super().__init__()
        self.lambda_mono = lambda_mono
        self.lambda_ord  = lambda_ord
        self.mse         = nn.MSELoss()
        self.ce          = nn.CrossEntropyLoss()
        self.thresholds = torch.tensor([0.001, 0.01, 0.033])


        # Probability thresholds for class assignment
        # Class 1: p < 0.002
        # Class 2: 0.002 <= p < 0.01
        # Class 3: 0.01  <= p < 0.035
        # Class 4: p >= 0.035
        self.thresholds = torch.tensor([0.002, 0.01, 0.035])

    def prob_to_class(self, p):
        thresholds = self.thresholds.to(p.device)
        c = torch.zeros(p.shape[0], dtype=torch.long, device=p.device)
        c[p >= thresholds[0]] = 1
        c[p >= thresholds[1]] = 2
        c[p >= thresholds[2]] = 3
        return c

    def forward(self, pred_curve, true_curve):
        """
        pred_curve: (B, 5) predicted exceedance probabilities
        true_curve: (B, 5) target exceedance probabilities
        """
        # 1. MSE loss on full curve
        loss_mse = self.mse(pred_curve, true_curve)

        # 2. Monotonicity penalty
        # pred_curve should be non-increasing
        # penalize any increases between consecutive depths
        diffs = pred_curve[:, 1:] - pred_curve[:, :-1]  # should be <= 0
        mono_violations = torch.clamp(diffs, min=0.0)
        loss_mono = mono_violations.mean()

        # 3. Ordinal classification loss on primary depth (0.2m)
        # Convert predicted probability at 0.2m to risk class
        pred_class_logits = self._prob_to_logits(pred_curve[:, 0])
        true_class = self.prob_to_class(true_curve[:, 0])
        loss_ord = self.ce(pred_class_logits, true_class)

        total = loss_mse + self.lambda_mono * loss_mono + self.lambda_ord * loss_ord
        return total, loss_mse, loss_mono, loss_ord

    def _prob_to_logits(self, p):
        """
        Convert scalar probability to 4-class logits.
        Uses distance to each threshold as proxy for confidence.
        """
        B = p.shape[0]
        thresholds = self.thresholds.to(p.device)

        # Soft assignment: logit for class k proportional to
        # proximity to that class's probability range
        t = torch.cat([
            torch.zeros(1, device=p.device),
            thresholds,
            torch.ones(1, device=p.device)
        ])  # boundaries: 0, 0.002, 0.01, 0.035, 1.0

        logits = torch.zeros(B, 4, device=p.device)
        for k in range(4):
            center = (t[k] + t[k+1]) / 2
            logits[:, k] = -torch.abs(p - center) * 100
        return logits

# ── 9. Evaluation: convert hazard curve back to risk classes ──────────
def hazard_to_class(hazard_curve_np, depth_idx=0):
    p = hazard_curve_np[:, depth_idx]
    classes = np.zeros(len(p), dtype=int)
    classes[p >= 0.001]  = 1   # Very Low
    classes[p >= 0.01]   = 2   # Low
    classes[p >= 0.033]  = 3   # Medium
    return classes

# ── 10. Build datasets ────────────────────────────────────────────────
print("\nBuilding datasets...")
print("Train dataset:")
train_ds = HazardPatchDataset(
    grid_s, hazard_grid_s, PATCH_SIZE,
    augment=True, val_fraction=0.2, is_val=False
)
print("Val dataset:")
val_ds = HazardPatchDataset(
    grid_s, hazard_grid_s, PATCH_SIZE,
    augment=False, val_fraction=0.2, is_val=True
)
print("Northumbria test dataset:")
test_ds = HazardPatchDataset(
    grid_n, hazard_grid_n, PATCH_SIZE,
    augment=False
)

from torch.utils.data import DataLoader
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=8, pin_memory=True, prefetch_factor=2
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE*2, shuffle=False,
    num_workers=8, pin_memory=True
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE*2, shuffle=False,
    num_workers=8, pin_memory=True
)

# ── 11. Model setup ───────────────────────────────────────────────────
model     = HazardCNN(N_CHANNELS, n_depths=N_DEPTHS).to(device)
criterion = HazardLoss(lambda_mono=0.1, lambda_ord=1.0)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=N_EPOCHS, pct_start=0.1
)
scaler = torch.amp.GradScaler('cuda')

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── 12. Training loop ─────────────────────────────────────────────────
print(f"\nTraining for {N_EPOCHS} epochs...")
print(f"{'Ep':>4} {'TrLoss':>8} {'TrAcc':>7} {'VlLoss':>8} {'VlAcc':>7} {'Time':>7}")
print("-" * 50)

best_val_acc = 0
best_state   = None

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()

    # ── Train ──────────────────────────────────────────────────────────
    model.train()
    tr_loss_sum, tr_correct, tr_total = 0, 0, 0

    for patches, hazard_true in train_loader:
        patches    = patches.to(device, non_blocking=True)
        hazard_true = hazard_true.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            hazard_pred = model(patches)
            loss, lmse, lmono, lord = criterion(hazard_pred, hazard_true)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Accuracy: convert predicted curve to class, compare to true class
        with torch.no_grad():
            pred_cls = hazard_to_class(hazard_pred.cpu().float().numpy())
            true_cls = hazard_to_class(hazard_true.cpu().float().numpy())
            tr_correct    += (pred_cls == true_cls).sum()
            tr_total      += len(pred_cls)
            tr_loss_sum   += loss.item() * len(pred_cls)

    tr_loss = tr_loss_sum / tr_total
    tr_acc  = tr_correct  / tr_total

    # ── Validate ───────────────────────────────────────────────────────
    model.eval()
    vl_loss_sum, vl_correct, vl_total = 0, 0, 0
    all_vl_pred, all_vl_true = [], []

    with torch.no_grad():
        for patches, hazard_true in val_loader:
            patches     = patches.to(device, non_blocking=True)
            hazard_true = hazard_true.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                hazard_pred = model(patches)
                loss, _, _, _ = criterion(hazard_pred, hazard_true)

            pred_cls = hazard_to_class(hazard_pred.cpu().float().numpy())
            true_cls = hazard_to_class(hazard_true.cpu().float().numpy())
            vl_correct    += (pred_cls == true_cls).sum()
            vl_total      += len(pred_cls)
            vl_loss_sum   += loss.item() * len(pred_cls)
            all_vl_pred.extend(pred_cls)
            all_vl_true.extend(true_cls)

    vl_loss = vl_loss_sum / vl_total
    vl_acc  = vl_correct  / vl_total
    elapsed = time.time() - t0

    marker = ' ◄' if vl_acc > best_val_acc else ''
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"{epoch:>4} {tr_loss:>8.4f} {tr_acc:>7.3f} "
          f"{vl_loss:>8.4f} {vl_acc:>7.3f} {elapsed:>6.1f}s{marker}")

# ── 13. Final evaluation ──────────────────────────────────────────────
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
model.eval()

def evaluate_loader(loader, label):
    all_pred, all_true = [], []
    all_curves_pred, all_curves_true = [], []

    with torch.no_grad():
        for patches, hazard_true in loader:
            patches     = patches.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                hazard_pred = model(patches)

            pred_cls = hazard_to_class(hazard_pred.cpu().float().numpy())
            true_cls = hazard_to_class(hazard_true.numpy())
            all_pred.extend(pred_cls)
            all_true.extend(true_cls)
            all_curves_pred.append(hazard_pred.cpu().float().numpy())
            all_curves_true.append(hazard_true.numpy())

    curves_pred = np.concatenate(all_curves_pred)
    curves_true = np.concatenate(all_curves_true)

    print(f"\n── Hazard CNN: {label} ──")
    print(f"Accuracy: {accuracy_score(all_true, all_pred):.3f}")
    print(classification_report(
        all_true, all_pred,
        target_names=['Very Low','Low','Medium','High']
    ))

    # Hazard curve quality metrics
    mse = np.mean((curves_pred - curves_true)**2)
    mae = np.mean(np.abs(curves_pred - curves_true))
    mono_violations = np.sum(np.diff(curves_pred, axis=1) > 0)
    print(f"Hazard curve MSE: {mse:.6f}")
    print(f"Hazard curve MAE: {mae:.6f}")
    print(f"Monotonicity violations: {mono_violations:,} "
          f"({mono_violations/curves_pred.size*100:.2f}%)")

    return accuracy_score(all_true, all_pred)

acc_sev  = evaluate_loader(val_loader,  "Severn val (seen)")
acc_nor  = evaluate_loader(test_loader, "Northumbria test (unseen)")

print("\n── Model progression ──")
print(f"XGBoost v4 (baseline) — Severn: 0.582 | Northumbria: 0.427")
print(f"CNN v2                — Severn: 0.504 | Northumbria: 0.362")
print(f"Hybrid CNN-GNN        — Severn: 0.459 | Northumbria: 0.285")
print(f"Hazard Curve CNN      — Severn: {acc_sev:.3f} | Northumbria: {acc_nor:.3f}")

torch.save(best_state, '/workspace/Flood-Risk/flood_hazard_cnn.pt')
print("\nSaved to /workspace/Flood-Risk/flood_hazard_cnn.pt")