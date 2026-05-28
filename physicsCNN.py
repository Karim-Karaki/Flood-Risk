import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy.spatial import cKDTree
from torch.utils.data import Dataset, DataLoader
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
TARGET_COL = 'risk_0_2m'

LAMBDA_FLOW = 0.5
LAMBDA_ELEV = 0.3
LAMBDA_ACC  = 0.3

FEATURE_COLS = [
    'dtm_zscore', 'log_flow_acc', 'imd', 'waw',
    'is_waterway', 'clc_type_clean',
    'tp_p99_zscore', 'max_rolling5_tp_zscore',
    'sro_p95_zscore', 'swvl1_min_zscore'
]
DTM_IDX      = FEATURE_COLS.index('dtm_zscore')
FLOW_ACC_IDX = FEATURE_COLS.index('log_flow_acc')
N_CHANNELS   = len(FEATURE_COLS)

# ── 2. Load data ──────────────────────────────────────────────────────
print("\nLoading datasets...")
ds_terrain_severn      = xr.open_dataset(DATA_DIR + 'Copy of Copy of flood_risk_terrain_severn.nc',      engine='netcdf4')
ds_terrain_northumbria = xr.open_dataset(DATA_DIR + 'Copy of Copy of flood_risk_terrain_northumbria.nc', engine='netcdf4')
ds_era5_severn         = xr.open_dataset(DATA_DIR + 'Copy of Copy of era5_land_severn.nc',               engine='netcdf4')
ds_era5_northumbria    = xr.open_dataset(DATA_DIR + 'Copy of Copy of era5_land_northumbria.nc',           engine='netcdf4')
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

print("Processing Severn...")
df_s = ds_terrain_severn.to_dataframe().reset_index()
df_s = filter_risk_pixels(df_s)
df_s = df_s[df_s['risk_0_2m'].isin([1.,2.,3.,4.])].copy()
df_s = engineer_terrain(df_s)
w_s  = engineer_weather(ds_era5_severn)
df_s = merge_weather(df_s, w_s)
df_s = add_weather_zscore(df_s, ['tp_p99','sro_p95','swvl1_min','max_rolling5_tp'])
df_s = df_s.reset_index(drop=True)
print(f"Severn: {len(df_s):,} pixels")

print("Processing Northumbria...")
df_n = ds_terrain_northumbria.to_dataframe().reset_index()
df_n = filter_risk_pixels(df_n)
df_n = engineer_terrain(df_n)
w_n  = engineer_weather(ds_era5_northumbria)
df_n = merge_weather(df_n, w_n)
df_n = add_weather_zscore(df_n, ['tp_p99','sro_p95','swvl1_min','max_rolling5_tp'])
df_n = df_n.reset_index(drop=True)
print(f"Northumbria: {len(df_n):,} pixels")

# ── 4. Build raster grid ──────────────────────────────────────────────
def df_to_grid(df, feature_cols, target_col, resolution=20):
    df = df.copy()
    df['yr'] = (df['y'] / resolution).round().astype(int)
    df['xr'] = (df['x'] / resolution).round().astype(int)
    mask = ~df.duplicated(subset=['yr','xr'])
    df   = df[mask].copy()

    yr_vals = np.sort(df['yr'].unique())[::-1]
    xr_vals = np.sort(df['xr'].unique())
    yr_to_i = {v: i for i, v in enumerate(yr_vals)}
    xr_to_j = {v: j for j, v in enumerate(xr_vals)}

    H, W, C = len(yr_vals), len(xr_vals), len(feature_cols)
    print(f"  Grid: {H} × {W} = {H*W:,} cells")

    feat_grid  = np.zeros((H, W, C), dtype=np.float32)
    label_grid = np.full((H, W), -1,  dtype=np.int8)

    feat_vals = df[feature_cols].fillna(0).values.astype(np.float32)
    labels    = df[target_col].values
    yr_idx    = df['yr'].map(yr_to_i).values
    xr_idx    = df['xr'].map(xr_to_j).values

    feat_grid[yr_idx, xr_idx, :] = feat_vals
    valid = ~np.isnan(labels) & np.isin(labels, [1,2,3,4])
    label_grid[yr_idx[valid], xr_idx[valid]] = (labels[valid] - 1).astype(np.int8)

    return feat_grid, label_grid

print("\nBuilding Severn grid...")
df_s_clean = df_s.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy().reset_index(drop=True)
grid_s, labels_s = df_to_grid(df_s_clean, FEATURE_COLS, TARGET_COL)

print("Building Northumbria grid...")
df_n_clean = df_n.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy().reset_index(drop=True)
grid_n, labels_n = df_to_grid(df_n_clean, FEATURE_COLS, TARGET_COL)

# ── 5. Physics-aware patch dataset ────────────────────────────────────
class PhysicsPatchDataset(Dataset):
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
        all_pos = list(zip(ys.tolist(), xs.tolist()))
        rng     = np.random.default_rng(seed)
        idx     = rng.permutation(len(all_pos))
        n_val   = int(len(all_pos) * val_fraction)

        if val_fraction > 0:
            selected = [all_pos[i] for i in (idx[:n_val] if is_val else idx[n_val:])]
        else:
            selected = all_pos

        self.positions = selected
        print(f"  Patches: {len(self.positions):,}")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        i, j  = self.positions[idx]
        half  = self.half

        patch = self.feat[i-half:i+half+1, j-half:j+half+1, :]
        patch = torch.from_numpy(patch.transpose(2, 0, 1).copy())
        label = int(self.label[i, j])

        center_dtm  = float(self.feat[i, j, DTM_IDX])
        center_flow = float(self.feat[i, j, FLOW_ACC_IDX])

        neighbor_dtm  = []
        neighbor_flow = []
        neighbor_lbl  = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i+di, j+dj
                neighbor_dtm.append(self.feat[ni, nj, DTM_IDX])
                neighbor_flow.append(self.feat[ni, nj, FLOW_ACC_IDX])
                neighbor_lbl.append(int(self.label[ni, nj]) if self.label[ni, nj] >= 0 else -1)

        if self.aug:
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[2])
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[1])

        return (
            patch,
            label,
            torch.tensor([center_dtm, center_flow], dtype=torch.float),
            torch.tensor(neighbor_dtm,  dtype=torch.float),
            torch.tensor(neighbor_flow, dtype=torch.float),
            torch.tensor(neighbor_lbl,  dtype=torch.long),
        )

# ── 6. Physics-informed loss ──────────────────────────────────────────
class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_flow=0.5, lambda_elev=0.3, lambda_acc=0.3):
        super().__init__()
        self.lambda_flow = lambda_flow
        self.lambda_elev = lambda_elev
        self.lambda_acc  = lambda_acc
        self.ce          = nn.CrossEntropyLoss()

    def forward(self, logits, labels,
                center_physics, neighbor_dtm, neighbor_flow, neighbor_lbl):
        loss_ce = self.ce(logits, labels)

        probs      = torch.softmax(logits, dim=1)
        class_vals = torch.tensor([0., 1., 2., 3.], device=logits.device)
        pred_risk  = (probs * class_vals).sum(dim=1)

        center_dtm  = center_physics[:, 0]
        center_flow = center_physics[:, 1]

        loss_flow = torch.tensor(0.0, device=logits.device)
        loss_elev = torch.tensor(0.0, device=logits.device)
        loss_acc  = torch.tensor(0.0, device=logits.device)
        n_valid   = 0

        for k in range(8):
            n_dtm  = neighbor_dtm[:, k]
            n_flow = neighbor_flow[:, k]
            n_lbl  = neighbor_lbl[:, k]

            valid = n_lbl >= 0
            if valid.sum() == 0:
                continue

            n_risk_val = n_lbl[valid].float()
            c_risk     = pred_risk[valid]
            c_dtm      = center_dtm[valid]
            c_flow     = center_flow[valid]
            n_dtm_v    = n_dtm[valid]
            n_flow_v   = n_flow[valid]

            # Flow: downstream neighbor should have >= risk
            downstream = n_flow_v > c_flow
            if downstream.sum() > 0:
                loss_flow = loss_flow + torch.clamp(
                    c_risk[downstream] - n_risk_val[downstream], min=0.0
                ).mean()

            # Elevation: lower pixel should have >= risk
            lower = n_dtm_v > c_dtm
            if lower.sum() > 0:
                loss_elev = loss_elev + torch.clamp(
                    n_risk_val[lower] - c_risk[lower], min=0.0
                ).mean()

            # Flow_acc: higher flow_acc should have >= risk
            high_acc = c_flow > n_flow_v
            if high_acc.sum() > 0:
                loss_acc = loss_acc + torch.clamp(
                    n_risk_val[high_acc] - c_risk[high_acc], min=0.0
                ).mean()

            n_valid += 1

        if n_valid > 0:
            loss_flow = loss_flow / n_valid
            loss_elev = loss_elev / n_valid
            loss_acc  = loss_acc  / n_valid

        total = (loss_ce
                 + self.lambda_flow * loss_flow
                 + self.lambda_elev * loss_elev
                 + self.lambda_acc  * loss_acc)

        return total, loss_ce, loss_flow, loss_elev, loss_acc

# ── 7. Model ──────────────────────────────────────────────────────────
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

class PhysicsCNN(nn.Module):
    def __init__(self, in_channels, n_classes=4):
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.gap(self.encoder(x)))

# ── 8. Build datasets ─────────────────────────────────────────────────
print("\nBuilding datasets...")
print("Train:")
train_ds = PhysicsPatchDataset(
    grid_s, labels_s, PATCH_SIZE,
    augment=True, val_fraction=0.2, is_val=False
)
print("Val:")
val_ds = PhysicsPatchDataset(
    grid_s, labels_s, PATCH_SIZE,
    augment=False, val_fraction=0.2, is_val=True
)
print("Northumbria test:")
test_ds = PhysicsPatchDataset(
    grid_n, labels_n, PATCH_SIZE,
    augment=False
)

def collate_fn(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.tensor([b[1] for b in batch], dtype=torch.long),
        torch.stack([b[2] for b in batch]),
        torch.stack([b[3] for b in batch]),
        torch.stack([b[4] for b in batch]),
        torch.stack([b[5] for b in batch]),
    )

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=8, pin_memory=True,
    prefetch_factor=2, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE*2, shuffle=False,
    num_workers=8, pin_memory=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE*2, shuffle=False,
    num_workers=8, pin_memory=True, collate_fn=collate_fn
)

# ── 9. Model setup ────────────────────────────────────────────────────
model     = PhysicsCNN(N_CHANNELS, n_classes=4).to(device)
criterion = PhysicsInformedLoss(
    lambda_flow=LAMBDA_FLOW,
    lambda_elev=LAMBDA_ELEV,
    lambda_acc=LAMBDA_ACC
)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=N_EPOCHS, pct_start=0.1
)
scaler = torch.amp.GradScaler('cuda')

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── 10. Training loop ─────────────────────────────────────────────────
print(f"\nTraining for {N_EPOCHS} epochs...")
print(f"{'Ep':>4} {'TrLoss':>8} {'CE':>7} {'Flow':>7} {'Elev':>7} {'TrAcc':>7} {'VlAcc':>7} {'VlF1':>7} {'Time':>7}")
print("-" * 75)

best_val_f1 = 0
best_state  = None

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()

    # ── Train ──────────────────────────────────────────────────────────
    model.train()
    tr_loss_sum = tr_ce_sum = tr_fl_sum = tr_el_sum = 0
    tr_correct  = tr_total  = 0

    for patches, labels, c_phys, n_dtm, n_flow, n_lbl in train_loader:
        patches = patches.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)
        c_phys  = c_phys.to(device,  non_blocking=True)
        n_dtm   = n_dtm.to(device,   non_blocking=True)
        n_flow  = n_flow.to(device,  non_blocking=True)
        n_lbl   = n_lbl.to(device,   non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(patches)
            loss, lce, lflow, lelev, lacc = criterion(
                logits, labels, c_phys, n_dtm, n_flow, n_lbl
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        B = len(labels)
        tr_loss_sum += loss.item()  * B
        tr_ce_sum   += lce.item()   * B
        tr_fl_sum   += lflow.item() * B
        tr_el_sum   += lelev.item() * B
        tr_correct  += (logits.detach().argmax(1) == labels).sum().item()
        tr_total    += B

    tr_loss = tr_loss_sum / tr_total
    tr_ce   = tr_ce_sum   / tr_total
    tr_fl   = tr_fl_sum   / tr_total
    tr_el   = tr_el_sum   / tr_total
    tr_acc  = tr_correct  / tr_total

    # ── Validate ───────────────────────────────────────────────────────
    model.eval()
    vl_preds_all = []
    vl_true_all  = []

    with torch.no_grad():
        for patches, labels, c_phys, n_dtm, n_flow, n_lbl in val_loader:
            patches = patches.to(device, non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(patches)
            vl_preds_all.extend(logits.argmax(1).cpu().numpy())
            vl_true_all.extend(labels.cpu().numpy())

    vl_acc = accuracy_score(vl_true_all, vl_preds_all)
    vl_f1  = f1_score(vl_true_all, vl_preds_all, average='weighted', zero_division=0)
    elapsed = time.time() - t0

    marker = ' ◄' if vl_f1 > best_val_f1 else ''
    if vl_f1 > best_val_f1:
        best_val_f1 = vl_f1
        best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"{epoch:>4} {tr_loss:>8.4f} {tr_ce:>7.4f} {tr_fl:>7.4f} "
          f"{tr_el:>7.4f} {tr_acc:>7.3f} {vl_acc:>7.3f} {vl_f1:>7.3f} {elapsed:>6.1f}s{marker}")

# ── 11. Final evaluation ──────────────────────────────────────────────
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
model.eval()

def evaluate_loader(loader, label):
    all_pred, all_true = [], []
    with torch.no_grad():
        for patches, labels, c_phys, n_dtm, n_flow, n_lbl in loader:
            patches = patches.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(patches)
            all_pred.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(labels.numpy())

    acc = accuracy_score(all_true, all_pred)
    f1  = f1_score(all_true, all_pred, average='weighted', zero_division=0)

    print(f"\n── Physics CNN: {label} ──")
    print(f"Accuracy: {acc:.3f} | Weighted F1: {f1:.3f}")
    print(classification_report(
        all_true, all_pred,
        target_names=['Very Low','Low','Medium','High'],
        zero_division=0
    ))
    return acc, f1

acc_sev, f1_sev = evaluate_loader(val_loader,  "Severn val (seen)")
acc_nor, f1_nor = evaluate_loader(test_loader, "Northumbria test (unseen)")

print("\n── Model progression (Weighted F1) ──")
print(f"XGBoost v4     — Severn: 0.56 | Northumbria: 0.37")
print(f"CNN v2         — Severn: 0.51 | Northumbria: 0.38")
print(f"Hybrid CNN-GNN — Severn: 0.40 | Northumbria: 0.22")
print(f"Physics CNN    — Severn: {f1_sev:.3f} | Northumbria: {f1_nor:.3f}")

torch.save(best_state, '/workspace/Flood-Risk/flood_physics_cnn.pt')
print("\nSaved to /workspace/Flood-Risk/flood_physics_cnn.pt")