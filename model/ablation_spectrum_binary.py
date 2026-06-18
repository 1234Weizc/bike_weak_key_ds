import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import time
import os
import warnings
warnings.filterwarnings('ignore')
# Fix cuDNN initialization error
import torch.backends.cudnn
# Disable cuDNN
torch.backends.cudnn.enabled = False

# =========================
# Device configuration
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =========================
# Global parameters
# =========================
num_bits = 12323

# Distances 1,2,...,6161
# If CSV contains distance=0, column 0 will be automatically removed
seq_len = num_bits // 2   # 6161

# Binary classification:
# 0 = random key
# 1 = m-gather weak key
num_classes = 1


# ============================================================
# [MOD-1] Ablation experiment config: change ablation_name to switch experiments
# ============================================================
# Options:
#   "baseline_res_tcn_dual"        : original spectrum + smooth spectrum + residual + dilation (main model)
#   "raw_only"                     : remove smooth channel, keep only original spectrum
#   "smooth_only"                  : remove original spectrum, keep only smooth spectrum
#   "no_residual"                  : remove residual connection, keep dual channel and dilation
#   "no_residual_raw_only"         : remove residual and smooth spectrum, i.e. plain TCN + original spectrum
#   "res_cnn_no_dilation"          : remove dilation, keep residual, i.e. Res-CNN
#   "plain_cnn_no_dilation_no_res" : remove dilation and residual, i.e. plain CNN
#
# Note: for fair comparison, except for the ablated factors, all other settings
# (train/val/test split, standardization, optimizer, epochs, patience) remain identical.
ablation_name = "plain_cnn_no_dilation_no_res"


def get_ablation_config(name):
    """
    [MOD-2] Each ablation experiment only changes the necessary factors.
    channel_mode:
        "dual"        -> original spectrum + smooth spectrum
        "raw"         -> original spectrum only
        "smooth"      -> smooth spectrum only
    use_residual:
        True/False     -> whether to use residual connection
    use_dilation:
        True/False     -> whether to use dilation=1,2,4,...; when False, all blocks use dilation=1
    norm_type:
        "bn"          -> BatchNorm1d, default
        "ln"          -> LayerNorm, if additional BN/LN comparison experiments are needed
    """
    configs = {
        "baseline_res_tcn_dual": {
            "channel_mode": "dual",
            "use_residual": True,
            "use_dilation": True,
            "norm_type": "bn",
        },
        "raw_only": {
            "channel_mode": "raw",
            "use_residual": True,
            "use_dilation": True,
            "norm_type": "bn",
        },
        "smooth_only": {
            "channel_mode": "smooth",
            "use_residual": True,
            "use_dilation": True,
            "norm_type": "bn",
        },
        "no_residual": {
            "channel_mode": "dual",
            "use_residual": False,
            "use_dilation": True,
            "norm_type": "bn",
        },
        "no_residual_raw_only": {
            "channel_mode": "raw",
            "use_residual": False,
            "use_dilation": True,
            "norm_type": "bn",
        },
        "res_cnn_no_dilation": {
            "channel_mode": "dual",
            "use_residual": True,
            "use_dilation": False,
            "norm_type": "bn",
        },
        "plain_cnn_no_dilation_no_res": {
            "channel_mode": "raw",
            "use_residual": False,
            "use_dilation": False,
            "norm_type": "bn",
        },
    }
    if name not in configs:
        raise ValueError(f"Unknown ablation_name={name}. Available: {list(configs.keys())}")
    return configs[name]


# =========================
# Dataset
# =========================
class SpectrumBinaryDataset(Dataset):
    def __init__(self, DS_data, labels, channel_mode="dual"):
        """
        DS_data: shape = [N, 6161]
        labels:  shape = [N]

        channel_mode:
            "dual"          : [raw, smooth]
            "raw"           : [raw]
            "smooth"        : [smooth]
            "triple"        : [raw, smooth, zscore]
            "raw_zscore"    : [raw, zscore]
            "smooth_zscore" : [smooth, zscore]
        """
        assert channel_mode in [
            "dual",
            "raw",
            "smooth",
            "triple",
            "raw_zscore",
            "smooth_zscore",
        ]

        self.DS_data = np.asarray(DS_data, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.channel_mode = channel_mode

    def __len__(self):
        return len(self.DS_data)

    @staticmethod
    def smooth_spectrum(x):
        """
        Light smoothing: 5-point weighted average
        """
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
        x_pad = np.pad(x, (2, 2), mode='edge')
        y = np.convolve(x_pad, kernel, mode='valid')
        return y.astype(np.float32)

    @staticmethod
    def zscore_spectrum(x, eps=1e-6):
        """
        Per-sample z-score normalization:
            z_i = (x_i - mean(x)) / (std(x) + eps)
        """
        mu = np.mean(x)
        std = np.std(x)
        return ((x - mu) / (std + eps)).astype(np.float32)

    def __getitem__(self, idx):
        x_raw = self.DS_data[idx].astype(np.float32)   # [6161]
        y = self.labels[idx]                           # scalar

        x_smooth = None
        x_zscore = None

        if self.channel_mode == "dual":
            x_smooth = self.smooth_spectrum(x_raw)
            x = np.stack([x_raw, x_smooth], axis=-1)        # [6161, 2]

        elif self.channel_mode == "raw":
            x = x_raw[:, np.newaxis]                       # [6161, 1]

        elif self.channel_mode == "smooth":
            x_smooth = self.smooth_spectrum(x_raw)
            x = x_smooth[:, np.newaxis]                    # [6161, 1]

        elif self.channel_mode == "triple":
            x_smooth = self.smooth_spectrum(x_raw)
            x_zscore = self.zscore_spectrum(x_raw)
            x = np.stack([x_raw, x_smooth, x_zscore], axis=-1)  # [6161, 3]

        elif self.channel_mode == "raw_zscore":
            x_zscore = self.zscore_spectrum(x_raw)
            x = np.stack([x_raw, x_zscore], axis=-1)       # [6161, 2]

        elif self.channel_mode == "smooth_zscore":
            x_smooth = self.smooth_spectrum(x_raw)
            x_zscore = self.zscore_spectrum(x_raw)
            x = np.stack([x_smooth, x_zscore], axis=-1)    # [6161, 2]

        else:
            raise ValueError(f"Unknown channel_mode: {self.channel_mode}")

        y = np.array([y], dtype=np.float32)                # [1]

        return torch.from_numpy(x), torch.from_numpy(y)

# =========================
# Read distance spectrum file
# =========================
def read_ds_file(how_many_to_read, filename, alternating_key_ds=True):
    """
    Read distance spectrum.

    alternating_key_ds=True:
        Corresponds to your original CSV format:
        Row 0 key
        Row 1 DS
        Row 2 key
        Row 3 DS
        ...

    alternating_key_ds=False:
        Each row is a distance spectrum DS.
    """
    DS = []
    still_todo = how_many_to_read

    with open(filename, 'r') as file:
        reader = csv.reader(file)

        for i, row in enumerate(reader):
            if alternating_key_ds:
                # Only read odd rows (DS)
                if i % 2 == 0:
                    continue

            values = [float(value.strip()) for value in row]
            DS.append(values)
            still_todo -= 1

            if still_todo == 0:
                break

    print(f"Read {how_many_to_read - still_todo} spectra from {filename}")

    DS_array = np.asarray(DS, dtype=np.float32)
    return DS_array


def crop_to_6161(DS):
    """
    Unify to length 6161.

    If input is 6162, assume column 0 is distance=0 and drop it.
    If input is already 6161, keep it unchanged.
    """
    if DS.shape[1] == seq_len + 1:
        print("Detected length 6162. Drop distance=0 column.")
        DS = DS[:, 1:]
    elif DS.shape[1] == seq_len:
        print("Detected length 6161. Keep unchanged.")
    else:
        raise ValueError(f"Unexpected DS length: {DS.shape[1]}, expected 6161 or 6162.")

    return DS.astype(np.float32)


# =========================
# standardizer
# =========================
def standardizer(data):
    """Min-max normalization"""
    normalized_data = np.zeros_like(data)
    
    for i in range(len(data)):
        min_val = np.min(data[i])
        max_val = np.max(data[i])
        if max_val - min_val > 0:
            normalized_data[i] = (data[i] - min_val) / (max_val - min_val)
        else:
            normalized_data[i] = data[i] * 0  # Set all values to 0
    
    return normalized_data

"""def fit_global_standardizer(X_train):
    # Compute global mean and std of training set
    mean = float(np.mean(X_train))
    std = float(np.std(X_train))

    if std < 1e-8:
        std = 1.0

    return mean, std


def apply_global_standardizer(X, mean, std):
    # Global mean/std standardization (commented out)
    return ((X - mean) / std).astype(np.float32)"""


# =========================
# Data splitting
# =========================
def split_two_class_data(random_DS, mgather_DS):
    """
    random_DS:  [10000, 6161]
    mgather_DS: [10000, 6161]

    Splitting method:
    random:
        0:8000      train
        8000:9000   val
        9000:10000  test

    m-gather:
        0:8000      train
        8000:9000   val
        9000:10000  test
    """
    assert random_DS.shape[0] >= 10000
    assert mgather_DS.shape[0] >= 10000

    random_DS = random_DS[:10000]
    mgather_DS = mgather_DS[:10000]

    y_random = np.zeros(10000, dtype=np.float32)
    y_mgather = np.ones(10000, dtype=np.float32)

    X_train = np.concatenate([random_DS[0:8000], mgather_DS[0:8000]], axis=0)
    y_train = np.concatenate([y_random[0:8000], y_mgather[0:8000]], axis=0)

    X_val = np.concatenate([random_DS[8000:9000], mgather_DS[8000:9000]], axis=0)
    y_val = np.concatenate([y_random[8000:9000], y_mgather[8000:9000]], axis=0)

    X_test = np.concatenate([random_DS[9000:10000], mgather_DS[9000:10000]], axis=0)
    y_test = np.concatenate([y_random[9000:10000], y_mgather[9000:10000]], axis=0)

    print("\nSplit done:")
    print(f"Train: {X_train.shape}, label mean = {y_train.mean():.4f}")
    print(f"Val:   {X_val.shape}, label mean = {y_val.mean():.4f}")
    print(f"Test:  {X_test.shape}, label mean = {y_test.mean():.4f}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def split_concat_ordered_data(all_DS):
    """
    If your data is in a single combined file:
        [0:10000]     random
        [10000:20000] m-gather

    Then use this function.
    """
    assert all_DS.shape[0] >= 20000

    random_DS = all_DS[0:10000]
    mgather_DS = all_DS[10000:20000]

    return split_two_class_data(random_DS, mgather_DS)


# =========================
# Model: unified implementation of Residual TCN / Res-CNN / plain CNN
# =========================
class ChannelLayerNorm1D(nn.Module):
    """
    [MOD-5] Optional LayerNorm: applies LN on channel dim C of Conv1d output [B, C, L].
    Default experiments still use BatchNorm1d; only set norm_type="ln" for extra BN vs LN experiments.
    """
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.transpose(1, 2)   # [B, L, C]
        x = self.ln(x)
        x = x.transpose(1, 2)   # [B, C, L]
        return x


def make_norm_1d(channels, norm_type="bn"):
    if norm_type == "bn":
        return nn.BatchNorm1d(channels)
    if norm_type == "ln":
        return ChannelLayerNorm1D(channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type: {norm_type}")


class ConvBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=7,
        dilation=1,
        dropout=0.1,
        use_residual=True,
        norm_type="bn",
    ):
        super().__init__()

        assert kernel_size % 2 == 1, "kernel_size should be odd."

        # [MOD-6] When use_dilation=False, dilation=1 is passed in; thus padding automatically degrades to plain CNN padding.
        padding = dilation * (kernel_size // 2)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.norm1 = make_norm_1d(out_channels, norm_type)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.norm2 = make_norm_1d(out_channels, norm_type)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # [MOD-7] Residual connection switch: when use_residual=False, shortcut is not used in forward.
        self.use_residual = use_residual
        if use_residual and in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        if self.use_residual:
            identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.use_residual:
            out = out + identity

        out = self.act(out)
        return out


class SpectrumCNNBinaryClassifier(nn.Module):
    """
    [MOD-8] Unified model:
        - use_residual=True,  use_dilation=True  -> Res-TCN
        - use_residual=False, use_dilation=True  -> plain TCN
        - use_residual=True,  use_dilation=False -> Res-CNN
        - use_residual=False, use_dilation=False -> plain CNN

    Input:  [B, 6161, C]
    Output: [B, 1] logits
    """
    def __init__(self, in_channels=2, use_residual=True, use_dilation=True, norm_type="bn"):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=33, padding=16),
            make_norm_1d(64, norm_type),
            nn.GELU()
        )

        dilations = [1, 2, 4, 8, 16, 32] if use_dilation else [1, 1, 1, 1, 1, 1]

        self.blocks = nn.Sequential(
            ConvBlock1D(64, 64, kernel_size=7, dilation=dilations[0], use_residual=use_residual, norm_type=norm_type),
            ConvBlock1D(64, 64, kernel_size=7, dilation=dilations[1], use_residual=use_residual, norm_type=norm_type),
            ConvBlock1D(64, 128, kernel_size=7, dilation=dilations[2], use_residual=use_residual, norm_type=norm_type),
            ConvBlock1D(128, 128, kernel_size=7, dilation=dilations[3], use_residual=use_residual, norm_type=norm_type),
            ConvBlock1D(128, 128, kernel_size=7, dilation=dilations[4], use_residual=use_residual, norm_type=norm_type),
            ConvBlock1D(128, 128, kernel_size=7, dilation=dilations[5], use_residual=use_residual, norm_type=norm_type),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: [B, L, C]
        x = x.transpose(1, 2)  # [B, C, L]

        x = self.stem(x)
        x = self.blocks(x)

        avg_feat = self.avg_pool(x).squeeze(-1)
        max_feat = self.max_pool(x).squeeze(-1)

        feat = torch.cat([avg_feat, max_feat], dim=1)
        logits = self.classifier(feat)
        return logits


# =========================
# Create model
# =========================
def get_model(model_name="spectrum_cnn", config=None):
    """
    [MOD-9] Model creation function accepts config to automatically adapt channels, residual, dilation.
    """
    if config is None:
        config = get_ablation_config(ablation_name)

    channel_mode = config["channel_mode"]
    in_channels = 2 if channel_mode == "dual" else 1

    if model_name == "spectrum_cnn":
        model = SpectrumCNNBinaryClassifier(
            in_channels=in_channels,
            use_residual=config["use_residual"],
            use_dilation=config["use_dilation"],
            norm_type=config.get("norm_type", "bn"),
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model.to(device)


# =========================
# Save / Load model
# =========================
def save_model(model, filename, model_name, config):
    output_dir = os.path.dirname(filename)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    # [MOD-10] Save config to avoid structure mismatch when loading single/dual channel models later.
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'config': config,
    }, filename)

    print(f"Model saved to {filename}")


def load_model(filename):
    checkpoint = torch.load(filename, map_location=device)
    model_name = checkpoint.get('model_name', 'spectrum_cnn')
    config = checkpoint.get('config', get_ablation_config(ablation_name))

    model = get_model(model_name, config=config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device)
    model.eval()

    print(f"Model loaded from {filename}, model_name = {model_name}, config = {config}")
    return model


# =========================
# Training and validation
# =========================
def evaluate_binary(model, data_loader, threshold=0.5):
    model.eval()

    all_probs = []
    all_labels = []

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(data)
            loss = criterion(logits, target)

            probs = torch.sigmoid(logits)

            total_loss += loss.item()
            batch_count += 1

            all_probs.append(probs.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0).reshape(-1)
    labels = np.concatenate(all_labels, axis=0).reshape(-1)

    preds = (probs >= threshold).astype(np.float32)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    avg_loss = total_loss / max(batch_count, 1)

    metrics = {
        "loss": avg_loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "probs": probs,
        "labels": labels
    }

    return metrics


def train_model(model, train_loader, val_loader=None, epochs=40, patience=6, min_delta=1e-4):
    """
    Binary classification training:
    - Uses BCEWithLogitsLoss
    - Model outputs logits, no Sigmoid in the model
    - Early stopping monitors val_loss
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_f1s = []

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(data)
            loss = criterion(logits, target)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if batch_idx % 20 == 0:
                print(
                    f"Epoch: {epoch+1}/{epochs}, "
                    f"Batch: {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}"
                )

        avg_train_loss = epoch_loss / max(batch_count, 1)
        train_losses.append(avg_train_loss)

        if val_loader is not None:
            val_metrics = evaluate_binary(model, val_loader, threshold=0.5)
            avg_val_loss = val_metrics["loss"]
            val_f1 = val_metrics["f1"]

            val_losses.append(avg_val_loss)
            val_f1s.append(val_f1)

            print(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss={avg_train_loss:.6f}, "
                f"Val Loss={avg_val_loss:.6f}, "
                f"Val Acc={val_metrics['acc']:.4f}, "
                f"Val F1={val_metrics['f1']:.4f}, "
                f"TP={val_metrics['tp']}, TN={val_metrics['tn']}, "
                f"FP={val_metrics['fp']}, FN={val_metrics['fn']}"
            )

            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print("Validation loss improved. Save current best model in memory.")
            else:
                patience_counter += 1
                print(f"No improvement. EarlyStop patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}")

    model.load_state_dict(best_model_state)

    if val_loader is not None:
        print(f"Restored best model with val_loss = {best_val_loss:.6f}")
    else:
        print("Training finished without validation loader.")

    return train_losses, val_losses, val_f1s


# =========================
# Save test set prediction results
# =========================
def save_binary_predictions(probs, labels, output_filename):
    output_dir = os.path.dirname(output_filename)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    with open(output_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "true_label", "pred_prob_mgather", "pred_label"])

        for i, (p, y) in enumerate(zip(probs, labels)):
            pred = 1 if p >= 0.5 else 0
            writer.writerow([i, int(y), float(p), int(pred)])

    print(f"Saved predictions to: {output_filename}")


# =========================
# Main function
# =========================
def main():
    # =========================
    # Parameter configuration
    # =========================

    # [MOD-11] Automatically derive experiment settings from ablation_name.
    config = get_ablation_config(ablation_name)
    model_name = "spectrum_cnn"

    print("\n================ Ablation Config ================")
    print(f"ablation_name = {ablation_name}")
    print(f"config        = {config}")
    print("=================================================\n")

    use_old_model = 0

    train_batch_size = 32
    val_batch_size = 32
    test_batch_size = 32

    epochs = 40
    patience = 6

    # Set to True if your CSV has alternating key/DS/key/DS rows
    # Set to False if each row is a DS
    alternating_key_ds = True

    # =========================
    # Case A: Two files
    # =========================
    random_file = r'....csv'
    mgather_file = r'....csv'

    random_DS = read_ds_file(
        how_many_to_read=10000,
        filename=random_file,
        alternating_key_ds=alternating_key_ds
    )

    mgather_DS = read_ds_file(
        how_many_to_read=10000,
        filename=mgather_file,
        alternating_key_ds=alternating_key_ds
    )

    random_DS = crop_to_6161(random_DS)
    mgather_DS = crop_to_6161(mgather_DS)

    X_train, y_train, X_val, y_val, X_test, y_test = split_two_class_data(
        random_DS,
        mgather_DS
    )
    # =========================
    # Standardization
    # =========================


    X_train = standardizer(X_train)
    X_val = standardizer(X_val)
    X_test = standardizer(X_test)

    # Global mean/std standardization (commented out)
    # mean, std = fit_global_standardizer(X_train)

    # X_train = apply_global_standardizer(X_train, mean, std)
    # X_val = apply_global_standardizer(X_val, mean, std)
    # X_test = apply_global_standardizer(X_test, mean, std)

    # print(f"\nNormalizer: mean={mean:.6f}, std={std:.6f}")


    # =========================
    # Build DataLoader
    # =========================
    # [MOD-12] The input channels of DataLoader are controlled by config["channel_mode"].
    channel_mode = config["channel_mode"]
    train_dataset = SpectrumBinaryDataset(X_train, y_train, channel_mode=channel_mode)
    val_dataset = SpectrumBinaryDataset(X_val, y_val, channel_mode=channel_mode)
    test_dataset = SpectrumBinaryDataset(X_test, y_test, channel_mode=channel_mode)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=(device.type == 'cuda')
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=(device.type == 'cuda')
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=(device.type == 'cuda')
    )

    # =========================
    # Model
    # =========================
    if use_old_model:
        model_filename = r'model\binary\....pth'
        model = load_model(model_filename)
    else:
        # [MOD-13] Model structure is controlled by config: channels, residual, dilation, normalization.
        model = get_model(model_name, config=config)

        print("\nStart training binary classifier...")
        train_losses, val_losses, val_f1s = train_model(
            model,
            train_loader,
            val_loader=val_loader,
            epochs=epochs,
            patience=patience
        )

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_filename = (
            f'model/binary/'
            f'my_binary_{ablation_name}_{timestamp}.pth'
        )

        save_model(model, model_filename, model_name, config)

    # =========================
    # Test set evaluation
    # =========================
    print("\nEvaluate on test set...")
    test_metrics = evaluate_binary(model, test_loader, threshold=0.5)

    print("\nTest set results:")
    print(f"Ablation  = {ablation_name}")
    print(f"Loss      = {test_metrics['loss']:.6f}")
    print(f"Accuracy  = {test_metrics['acc']:.4f}")
    print(f"Precision = {test_metrics['precision']:.4f}")
    print(f"Recall    = {test_metrics['recall']:.4f}")
    print(f"F1        = {test_metrics['f1']:.4f}")
    predict_positive_rate = (test_metrics['tp'] + test_metrics['fp']) / 2000
    print("\nConfusion Matrix:")
    print(f"TP = {test_metrics['tp']}   FP = {test_metrics['fp']}")
    print(f"FN = {test_metrics['fn']}   TN = {test_metrics['tn']}")
    print(f"predict_positive_rate = {predict_positive_rate}") # higher is better
    # =========================
    # Save prediction probabilities
    # =========================
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    pred_output_filename = (
        f'a_ toy_recontruction_of_lab/model/binary/'
        f'test_predictions_{ablation_name}_{timestamp}.csv'
    )

    save_binary_predictions(
        probs=test_metrics["probs"],
        labels=test_metrics["labels"],
        output_filename=pred_output_filename
    )

    probs = test_metrics["probs"]
    labels = test_metrics["labels"]

    print("prob random mean =", probs[labels == 0].mean())
    print("prob random max  =", probs[labels == 0].max())
    print("prob weak mean   =", probs[labels == 1].mean())
    print("prob weak min    =", probs[labels == 1].min())

    print(f"p_true = {test_metrics['recall'] / predict_positive_rate}") # lower is better

if __name__ == "__main__":
    main()
