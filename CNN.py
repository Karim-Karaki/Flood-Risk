import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import time

# ── 0. Device check ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── 1. Config ─────────────────────────────────────────────────────────
PATCH_SIZE  = 11        # 11x11 pixels = 220m x 220m neighborhood
HALF        = PATCH_SIZE // 2
BATCH_SIZE  = 2048      # large batch for 4090
N_EPOCHS    = 30
LR          = 1e-3
TARGET_COL  = 'risk_0_2m'

FEATURE_COLS = [
    'dtm_zscore', 'log_flow_acc', 'imd', 'waw',
    'is_waterway', 'clc_type_clean',
    'tp_p99_zscore', 'max_rolling5_tp_zscore',
    'sro_p95_zscore', 'swvl1_min_zscore'
]
N_CHANNELS = len(FEATURE_COLS)
print(f"Features ({N_CHANNELS}): {FEATURE_COLS}")

# ── 2. Build raster grid from dataframe ───────────────────────────────
def df_to_grid(df, feature_cols, target_col, resolution=20):
    """
    Convert flat dataframe to 2D raster grids.
    Returns:
        feature_grid : (H, W, C) float32
        label_grid   : (H, W)    int8  — -1 = no data
        row_coords   : sorted y values (descending)
        col_coords   : sorted x values (ascending)
    """
    df = df.copy()
    
    # Round coordinates to nearest grid cell
    df['yr'] = (df['y'] / resolution).round().astype(int)
    df['xr'] = (df['x'] / resolution).round().astype(int)
    df = df.drop_duplicates(subset=['yr', 'xr'])
    
    # Build index maps
    yr_vals = np.sort(df['yr'].unique())[::-1]   # north → south
    xr_vals = np.sort(df['xr'].unique())          # west  → east
    yr_to_i = {v: i for i, v in enumerate(yr_vals)}
    xr_to_j = {v: j for j, v in enumerate(xr_vals)}
    
    H, W, C = len(yr_vals), len(xr_vals), len(feature_cols)
    print(f"  Grid: {H} rows × {W} cols = {H*W:,} cells | {C} channels")
    
    feat_grid  = np.zeros((H, W, C), dtype=np.float32)
    label_grid = np.full((H, W), -1,  dtype=np.int8)
    
    # Fill grids
    feat_vals = df[feature_cols].values.astype(np.float32)
    labels    = df[target_col].values
    yr_idx    = df['yr'].map(yr_to_i).values
    xr_idx    = df['xr'].map(xr_to_j).values
    
    for k in range(len(df)):
        i, j = yr_idx[k], xr_idx[k]
        feat_grid[i, j, :] = feat_vals[k]
        lbl = labels[k]
        if not np.isnan(lbl) and lbl in [1, 2, 3, 4]:
            label_grid[i, j] = int(lbl) - 1   # 0-indexed
    
    return feat_grid, label_grid, yr_vals, xr_vals

# ── 3. Patch Dataset ───────────────────────────────────────────────────
class FloodPatchDataset(Dataset):
    """
    Extracts PATCH_SIZE × PATCH_SIZE patches centered on valid risk pixels.
    Input  : (C, P, P) tensor
    Target : scalar class 0-3
    """
    def __init__(self, feat_grid, label_grid, patch_size=11, augment=False):
        self.feat  = feat_grid    # (H, W, C)
        self.label = label_grid   # (H, W)
        self.P     = patch_size
        self.half  = patch_size // 2
        self.aug   = augment
        H, W       = label_grid.shape
        
        # Collect valid center pixel positions (has a label, not edge)
        ys, xs = np.where(
            (label_grid >= 0) &
            (np.arange(H)[:, None] >= self.half) &
            (np.arange(H)[:, None] <  H - self.half) &
            (np.arange(W)[None, :] >= self.half) &
            (np.arange(W)[None, :] <  W - self.half)
        )
        self.positions = list(zip(ys.tolist(), xs.tolist()))
        print(f"  Valid patch centers: {len(self.positions):,}")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        i, j    = self.positions[idx]
        h, half = self.P, self.half
        
        # Extract patch (P, P, C) → (C, P, P)
        patch = self.feat[i-half:i+half+1, j-half:j+half+1, :]
        patch = torch.from_numpy(patch.transpose(2, 0, 1))  # (C, P, P)
        label = int(self.label[i, j])
        
        # Light augmentation — random horizontal/vertical flip
        if self.aug:
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[2])   # horizontal
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[1])   # vertical
        
        return patch, label

# ── 4. CNN Architecture ───────────────────────────────────────────────
class FloodCNN(nn.Module):
    """
    Lightweight CNN for flood risk classification.
    Input : (B, C, P, P)
    Output: (B, 4) logits
    
    Architecture inspired by Kabir et al. (2020) and Gao et al. (2024):
    - Progressive feature extraction with increasing filters
    - BatchNorm for training stability
    - Dropout for regularization
    - Global average pooling before classifier
    """
    def __init__(self, in_channels, n_classes=4, patch_size=11):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 11 → 5
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 5 → 2
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),     # 2 → 1
        )
        
        # Global average pooling → flatten
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# ── 5. Training function ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for patches, labels in loader:
        patches = patches.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        logits = model(patches)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
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
            
            logits = model(patches)
            loss   = criterion(logits, labels)
            
            total_loss += loss.item() * len(labels)
            preds       = logits.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += len(labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / total, correct / total, all_preds, all_labels

# ── 6. Build Severn grid ──────────────────────────────────────────────
print("\nBuilding Severn grid...")
df_s_clean = df_s_merged.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
df_s_clean = df_s_clean[df_s_clean[TARGET_COL].isin([1, 2, 3, 4])]
print(f"Severn clean pixels: {len(df_s_clean):,}")

grid_s, labels_s, yr_s, xr_s = df_to_grid(df_s_clean, FEATURE_COLS, TARGET_COL)

# ── 7. Train/val split — spatial blocks ───────────────────────────────
# Hold out southern 20% of rows for validation
H_s = grid_s.shape[0]
split_row = int(H_s * 0.8)

grid_train  = grid_s[:split_row, :, :]
labels_train = labels_s[:split_row, :]
grid_val    = grid_s[split_row:, :, :]
labels_val  = labels_s[split_row:, :]

print(f"\nSpatial train/val split:")
print(f"  Train rows: {split_row} | Val rows: {H_s - split_row}")
print(f"  Train valid pixels: {(labels_train >= 0).sum():,}")
print(f"  Val valid pixels:   {(labels_val >= 0).sum():,}")

# ── 8. Datasets and loaders ───────────────────────────────────────────
print("\nBuilding datasets...")
train_ds = FloodPatchDataset(grid_train, labels_train, PATCH_SIZE, augment=True)
val_ds   = FloodPatchDataset(grid_val,   labels_val,   PATCH_SIZE, augment=False)

# Class weights for imbalanced classes
label_counts = Counter([train_ds.label[i][j] 
                        for i, j in train_ds.positions])
total = sum(label_counts.values())
class_weights = torch.tensor(
    [total / (4 * label_counts.get(c, 1)) for c in range(4)],
    dtype=torch.float32
).to(device)
print(f"\nClass weights: {class_weights.cpu().numpy().round(3)}")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, prefetch_factor=2
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE*2, shuffle=False,
    num_workers=4, pin_memory=True
)

# ── 9. Model, optimizer, scheduler ───────────────────────────────────
model     = FloodCNN(N_CHANNELS, n_classes=4, patch_size=PATCH_SIZE).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,}")

# ── 10. Training loop ─────────────────────────────────────────────────
print(f"\nTraining on {device} for {N_EPOCHS} epochs...")
print(f"{'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>10} {'ValLoss':>10} {'ValAcc':>10} {'Time':>8}")
print("-" * 58)

best_val_acc = 0
best_state   = None
history      = []

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    vl_loss, vl_acc, vl_preds, vl_labels = eval_epoch(model, val_loader, criterion, device)
    scheduler.step()
    
    elapsed = time.time() - t0
    history.append({
        'epoch': epoch, 'tr_loss': tr_loss, 'tr_acc': tr_acc,
        'vl_loss': vl_loss, 'vl_acc': vl_acc
    })
    
    marker = ' ◄ best' if vl_acc > best_val_acc else ''
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    print(f"{epoch:>5} {tr_loss:>10.4f} {tr_acc:>10.3f} {vl_loss:>10.4f} {vl_acc:>10.3f} {elapsed:>7.1f}s{marker}")

print(f"\nBest val accuracy: {best_val_acc:.3f}")

# ── 11. Full evaluation on val set ────────────────────────────────────
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
_, _, vl_preds, vl_labels = eval_epoch(model, val_loader, criterion, device)

print("\n── CNN: Severn spatial val (seen, southern region) ──")
print(f"Accuracy: {accuracy_score(vl_labels, vl_preds):.3f}")
print(classification_report(vl_labels, vl_preds,
      target_names=['Very Low', 'Low', 'Medium', 'High']))