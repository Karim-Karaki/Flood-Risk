import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score
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
TARGET_COL = 'risk_1_2m'

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

# ── 2. Metrics helper ─────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, label):
    """
    Compute and print metrics table matching competition format:
    QWK | macro_f1 | weighted_f1 | f1_class_0..3 | N
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    N      = len(y_true)

    qwk      = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    macro_f1 = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    wf1      = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    acc      = accuracy_score(y_true, y_pred)
    per_cls  = f1_score(y_true, y_pred, average=None,       zero_division=0, labels=[0,1,2,3])

    # Table header
    sep  = "| --------- | ------ | -------- | ----------- | ---------- | ---------- | ---------- | ---------- | --------- |"
    hdr  = "| metric    | qwk    | macro_f1 | weighted_f1 | f1_class_0 | f1_class_1 | f1_class_2 | f1_class_3 | N         |"

    print(f"\n=== {label} ===")
    print(hdr)
    print(sep)
    print(f"| {TARGET_COL:<9} "
          f"| {qwk:.4f} "
          f"| {macro_f1:.4f}   "
          f"| {wf1:.4f}      "
          f"| {per_cls[0]:.4f}     "
          f"| {per_cls[1]:.4f}     "
          f"| {per_cls[2]:.4f}     "
          f"| {per_cls[3]:.4f}     "
          f"| {N:<9} |")
    print(sep)

    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(
        y_true, y_pred,
        target_names=['Very Low','Low','Medium','High'],
        zero_division=0
    ))
    return qwk, macro_f1, wf1, per_cls, acc

# ── 3. Load data ──────────────────────────────────────────────────────
print("\nLoading datasets...")
ds_terrain_severn      = xr.open_dataset(DATA_DIR + 'Copy of Copy of flood_risk_terrain_severn.nc',      engine='netcdf4')
ds_terrain_northumbria = xr.open_dataset(DATA_DIR + 'Copy of Copy of flood_risk_terrain_northumbria.nc', engine='netcdf4')
ds_era5_severn         = xr.open_dataset(DATA_DIR + 'Copy of Copy of era5_land_severn.nc',               engine='netcdf4')
ds_era5_northumbria    = xr.open_dataset(DATA_DIR + 'Copy of Copy of era5_land_northumbria.nc',           engine='netcdf4')
print("Loaded.")

# ── 4. Feature engineering ────────────────────────────────────────────
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

# ── 5. Build raster grid ──────────────────────────────────────────────
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

# ── 6. Physics-aware patch dataset ────────────────────────────────────
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

        neighbor_dtm, neighbor_flow, neighbor_lbl = [], [], []
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

# ── 7. Physics-informed loss ──────────────────────────────────────────
class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_flow=0.5, lambda_elev=0.3, lambda_acc=0.3):
        super().__init__()
        self.lambda_flow = lambda_flow
        self.lambda_elev = lambda_elev
        self.lambda_acc  = lambda_acc
        self.ce          = nn.CrossEntropyLoss()

    def forward(self, logits, labels,
                center_physics, neighbor_dtm, neighbor_flow, neighbor_lbl):
        loss_ce    = self.ce(logits, labels)
        probs      = torch.softmax(logits, dim=1)
        class_vals = torch.tensor([0.,1.,2.,3.], device=logits.device)
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
            valid  = n_lbl >= 0
            if valid.sum() == 0:
                continue

            n_risk_val = n_lbl[valid].float()
            c_risk     = pred_risk[valid]
            c_dtm      = center_dtm[valid]
            c_flow     = center_flow[valid]
            n_dtm_v    = n_dtm[valid]
            n_flow_v   = n_flow[valid]

            downstream = n_flow_v > c_flow
            if downstream.sum() > 0:
                loss_flow = loss_flow + torch.clamp(
                    c_risk[downstream] - n_risk_val[downstream], min=0.0).mean()

            lower = n_dtm_v > c_dtm
            if lower.sum() > 0:
                loss_elev = loss_elev + torch.clamp(
                    n_risk_val[lower] - c_risk[lower], min=0.0).mean()

            high_acc = c_flow > n_flow_v
            if high_acc.sum() > 0:
                loss_acc = loss_acc + torch.clamp(
                    n_risk_val[high_acc] - c_risk[high_acc], min=0.0).mean()

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

# ── 8. Model ──────────────────────────────────────────────────────────
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

# ── 9. Build datasets ─────────────────────────────────────────────────
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

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,   shuffle=True,
                          num_workers=8, pin_memory=True,
                          prefetch_factor=2, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False,
                          num_workers=8, pin_memory=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, shuffle=False,
                          num_workers=8, pin_memory=True, collate_fn=collate_fn)

# ── 10. Model setup ───────────────────────────────────────────────────
model     = PhysicsCNN(N_CHANNELS, n_classes=4).to(device)
criterion = PhysicsInformedLoss(LAMBDA_FLOW, LAMBDA_ELEV, LAMBDA_ACC)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=N_EPOCHS, pct_start=0.1
)
scaler = torch.amp.GradScaler('cuda')

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── 11. Training loop ─────────────────────────────────────────────────
print(f"\nTraining for {N_EPOCHS} epochs...")
print(f"{'Ep':>4} {'TrLoss':>8} {'CE':>7} {'Flow':>7} {'Elev':>7} {'TrAcc':>7} {'VlAcc':>7} {'VlF1':>7} {'VlQWK':>7} {'Time':>7}")
print("-" * 80)

best_val_qwk = 0
best_state   = None

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()

    # Train
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
                logits, labels, c_phys, n_dtm, n_flow, n_lbl)

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

    # Validate
    model.eval()
    vl_preds_all, vl_true_all = [], []

    with torch.no_grad():
        for patches, labels, c_phys, n_dtm, n_flow, n_lbl in val_loader:
            patches = patches.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(patches)
            vl_preds_all.extend(logits.argmax(1).cpu().numpy())
            vl_true_all.extend(labels.numpy())

    vl_acc = accuracy_score(vl_true_all, vl_preds_all)
    vl_f1  = f1_score(vl_true_all, vl_preds_all, average='weighted', zero_division=0)
    vl_qwk = cohen_kappa_score(vl_true_all, vl_preds_all, weights='quadratic')
    elapsed = time.time() - t0

    marker = ' ◄' if vl_qwk > best_val_qwk else ''
    if vl_qwk > best_val_qwk:
        best_val_qwk = vl_qwk
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"{epoch:>4} {tr_loss:>8.4f} {tr_ce:>7.4f} {tr_fl:>7.4f} "
          f"{tr_el:>7.4f} {tr_acc:>7.3f} {vl_acc:>7.3f} "
          f"{vl_f1:>7.3f} {vl_qwk:>7.3f} {elapsed:>6.1f}s{marker}")

# ── 12. Final evaluation ──────────────────────────────────────────────
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
model.eval()

def run_inference(loader):
    all_pred, all_true = [], []
    with torch.no_grad():
        for patches, labels, c_phys, n_dtm, n_flow, n_lbl in loader:
            patches = patches.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(patches)
            all_pred.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(labels.numpy())
    return np.array(all_true), np.array(all_pred)

sev_true, sev_pred = run_inference(val_loader)
nor_true, nor_pred = run_inference(test_loader)

qwk_sev, mf1_sev, wf1_sev, pc_sev, acc_sev = compute_metrics(sev_true, sev_pred, "Severn val (seen)")
qwk_nor, mf1_nor, wf1_nor, pc_nor, acc_nor = compute_metrics(nor_true, nor_pred, "Northumbria test (unseen)")

# Comparison table
print("\n=== Model progression ===")
hdr = "| model            | region       | qwk    | macro_f1 | weighted_f1 | f1_c0  | f1_c1  | f1_c2  | f1_c3  |"
sep = "| ---------------- | ------------ | ------ | -------- | ----------- | ------ | ------ | ------ | ------ |"
print(hdr)
print(sep)
rows = [
    ("XGBoost v4",    "Severn",       "~0.52", "~0.43", "0.56", "-", "-", "-", "-"),
    ("XGBoost v4",    "Northumbria",  "~0.35", "~0.28", "0.37", "-", "-", "-", "-"),
    ("CNN v2",        "Severn",       "~0.46", "~0.46", "0.51", "-", "-", "-", "-"),
    ("CNN v2",        "Northumbria",  "~0.28", "~0.37", "0.38", "-", "-", "-", "-"),
]
for r in rows:
    print(f"| {r[0]:<16} | {r[1]:<12} | {r[2]:<6} | {r[3]:<8} | {r[4]:<11} | {r[5]:<6} | {r[6]:<6} | {r[7]:<6} | {r[8]:<6} |")
print(f"| {'Physics CNN':<16} | {'Severn':<12} | {qwk_sev:.4f} | {mf1_sev:.4f}   | {wf1_sev:.4f}      | {pc_sev[0]:.4f} | {pc_sev[1]:.4f} | {pc_sev[2]:.4f} | {pc_sev[3]:.4f} |")
print(f"| {'Physics CNN':<16} | {'Northumbria':<12} | {qwk_nor:.4f} | {mf1_nor:.4f}   | {wf1_nor:.4f}      | {pc_nor[0]:.4f} | {pc_nor[1]:.4f} | {pc_nor[2]:.4f} | {pc_nor[3]:.4f} |")
print(sep)

# ── 13. Visual maps ───────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

RISK_CMAP   = mcolors.ListedColormap(['#2ecc71','#f1c40f','#e67e22','#e74c3c'])
RISK_BOUNDS = [-0.5, 0.5, 1.5, 2.5, 3.5]
RISK_NORM   = mcolors.BoundaryNorm(RISK_BOUNDS, RISK_CMAP.N)
RISK_LABELS = ['Very Low','Low','Medium','High']
OUT_DIR     = '/workspace/Flood-Risk/'

def get_pred_map(model, feat_grid, label_grid, device,
                 patch_size=11, batch_size=8192):
    half     = patch_size // 2
    H, W, C  = feat_grid.shape
    pred_map = np.full((H, W), -1, dtype=np.int8)

    ys, xs = np.where(
        (label_grid >= 0) &
        (np.arange(H)[:, None] >= half) &
        (np.arange(H)[:, None] <  H - half) &
        (np.arange(W)[None, :] >= half) &
        (np.arange(W)[None, :] <  W - half)
    )
    positions = list(zip(ys.tolist(), xs.tolist()))
    print(f"  Inference on {len(positions):,} pixels...")

    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            batch_pos = positions[start:start+batch_size]
            patches   = np.stack([
                feat_grid[i-half:i+half+1, j-half:j+half+1, :]
                .transpose(2,0,1)
                for i,j in batch_pos
            ])
            with torch.amp.autocast('cuda'):
                logits = model(torch.from_numpy(patches).to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())

    for (i,j), p in zip(positions, all_preds):
        pred_map[i,j] = p
    return pred_map

def plot_comparison(true_grid, pred_grid, region_name,
                    y_true_flat, y_pred_flat, save_path):
    true_show = np.where(true_grid >= 0, true_grid, np.nan).astype(float)
    pred_show = np.where(pred_grid >= 0, pred_grid, np.nan).astype(float)
    valid     = (true_grid >= 0) & (pred_grid >= 0)
    diff      = np.where(valid, (pred_grid == true_grid).astype(float), np.nan)

    # Metrics
    qwk  = cohen_kappa_score(y_true_flat, y_pred_flat, weights='quadratic')
    mf1  = f1_score(y_true_flat, y_pred_flat, average='macro',    zero_division=0)
    wf1  = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    acc  = accuracy_score(y_true_flat, y_pred_flat)
    pcls = f1_score(y_true_flat, y_pred_flat, average=None, zero_division=0, labels=[0,1,2,3])

    fig  = plt.figure(figsize=(22, 10))
    fig.suptitle(f'Physics CNN — {region_name}', fontsize=16, fontweight='bold')

    # Layout: 3 maps on top, metrics table on bottom
    gs   = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.35, wspace=0.1)

    ax0  = fig.add_subplot(gs[0, 0])
    ax1  = fig.add_subplot(gs[0, 1])
    ax2  = fig.add_subplot(gs[0, 2])
    ax_t = fig.add_subplot(gs[1, :])

    # Ground truth map
    ax0.imshow(true_show, cmap=RISK_CMAP, norm=RISK_NORM,
               interpolation='nearest', aspect='auto')
    ax0.set_title('Ground Truth', fontsize=13, fontweight='bold')
    ax0.axis('off')

    # Prediction map
    ax1.imshow(pred_show, cmap=RISK_CMAP, norm=RISK_NORM,
               interpolation='nearest', aspect='auto')
    ax1.set_title('Physics CNN Prediction', fontsize=13, fontweight='bold')
    ax1.axis('off')

    # Correct/wrong map
    diff_cmap = mcolors.ListedColormap(['#e74c3c','#2ecc71'])
    ax2.imshow(diff, cmap=diff_cmap, vmin=0, vmax=1,
               interpolation='nearest', aspect='auto')
    ax2.set_title('Correct (green) / Wrong (red)', fontsize=13, fontweight='bold')
    ax2.axis('off')

    # Legend for risk maps
    legend_patches = [
        Patch(color=RISK_CMAP(i/3), label=RISK_LABELS[i])
        for i in range(4)
    ]
    ax0.legend(handles=legend_patches, loc='lower left',
               fontsize=8, framealpha=0.8)

    # Metrics table
    ax_t.axis('off')
    col_labels = ['Region','Accuracy','QWK','Macro F1','Weighted F1',
                  'F1 Very Low','F1 Low','F1 Medium','F1 High']
    row_data   = [[
        region_name,
        f'{acc:.4f}',
        f'{qwk:.4f}',
        f'{mf1:.4f}',
        f'{wf1:.4f}',
        f'{pcls[0]:.4f}',
        f'{pcls[1]:.4f}',
        f'{pcls[2]:.4f}',
        f'{pcls[3]:.4f}',
    ]]
    tbl = ax_t.table(
        cellText=row_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    # Color header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Color metric cells by performance
    colors_qwk = ['#e74c3c','#e67e22','#f1c40f','#2ecc71']
    thresholds  = [0.3, 0.5, 0.65]
    def perf_color(val):
        if val < thresholds[0]: return '#e74c3c'
        if val < thresholds[1]: return '#e67e22'
        if val < thresholds[2]: return '#f1c40f'
        return '#2ecc71'

    metric_cols = [1,2,3,4,5,6,7,8]
    metric_vals = [acc, qwk, mf1, wf1, pcls[0], pcls[1], pcls[2], pcls[3]]
    for j, v in zip(metric_cols, metric_vals):
        tbl[1, j].set_facecolor(perf_color(float(v)))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_class_distribution(true_grid, pred_grid, region_name,
                             y_true_flat, y_pred_flat, save_path):
    colors = ['#2ecc71','#f1c40f','#e67e22','#e74c3c']
    x      = np.arange(4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Class Distribution — {region_name}', fontsize=14, fontweight='bold')

    for ax, data, lbl in zip(axes,
                              [y_true_flat, y_pred_flat],
                              ['Ground Truth','Physics CNN Prediction']):
        counts = [(data == c).sum() for c in range(4)]
        total  = sum(counts)
        pcts   = [c/total*100 for c in counts]
        bars   = ax.bar(x, pcts, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(RISK_LABELS, fontsize=11)
        ax.set_ylabel('% of pixels', fontsize=11)
        ax.set_title(lbl, fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(pcts) * 1.25)
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

# Generate maps
print("\nGenerating prediction maps...")

print("Severn inference...")
pred_map_s = get_pred_map(model, grid_s, labels_s, device)

print("Northumbria inference...")
pred_map_n = get_pred_map(model, grid_n, labels_n, device)

valid_s = (labels_s >= 0) & (pred_map_s >= 0)
valid_n = (labels_n >= 0) & (pred_map_n >= 0)

print("Plotting Severn...")
plot_comparison(
    labels_s, pred_map_s,
    'Severn (Seen Region)',
    labels_s[valid_s], pred_map_s[valid_s],
    OUT_DIR + 'map_severn_physics_cnn.png'
)
plot_class_distribution(
    labels_s, pred_map_s,
    'Severn (Seen Region)',
    labels_s[valid_s], pred_map_s[valid_s],
    OUT_DIR + 'dist_severn_physics_cnn.png'
)

print("Plotting Northumbria...")
plot_comparison(
    labels_n, pred_map_n,
    'Northumbria (Unseen Region)',
    labels_n[valid_n], pred_map_n[valid_n],
    OUT_DIR + 'map_northumbria_physics_cnn.png'
)
plot_class_distribution(
    labels_n, pred_map_n,
    'Northumbria (Unseen Region)',
    labels_n[valid_n], pred_map_n[valid_n],
    OUT_DIR + 'dist_northumbria_physics_cnn.png'
)

torch.save(best_state, OUT_DIR + 'flood_physics_cnn.pt')

print("\n── Output files ──")
print(f"  map_severn_physics_cnn.png")
print(f"  map_northumbria_physics_cnn.png")
print(f"  dist_severn_physics_cnn.png")
print(f"  dist_northumbria_physics_cnn.png")
print(f"  flood_physics_cnn.pt")