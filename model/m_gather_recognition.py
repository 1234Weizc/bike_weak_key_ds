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


# =========================
# Dataset
# =========================
class SpectrumBinaryDataset(Dataset):
    def __init__(self, DS_data, labels, use_dual_channel=False):
        """
        DS_data: shape = [N, 6161]
        labels:  shape = [N]
        """
        self.DS_data = np.asarray(DS_data, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.use_dual_channel = use_dual_channel

    def __len__(self):
        return len(self.DS_data)

    @staticmethod
    def smooth_spectrum(x):
        """
        Slightly smooth
        """
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
        x_pad = np.pad(x, (2, 2), mode='edge')
        y = np.convolve(x_pad, kernel, mode='valid')
        return y.astype(np.float32)

    def __getitem__(self, idx):
        x = self.DS_data[idx]   # [6161]
        y = self.labels[idx]    # scalar

        if self.use_dual_channel:
            x_smooth = self.smooth_spectrum(x)
            x = np.stack([x, x_smooth], axis=-1)  # [6161, 2]
        else:
            x = x[:, np.newaxis]                  # [6161, 1]

        y = np.array([y], dtype=np.float32)        # [1]

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
# Standardization
# =========================
def fit_global_standardizer(X_train):
    """
    Fit global mean/std using training set.
    Note: do not fit on val/test to avoid data leakage.
    """
    mean = float(np.mean(X_train))
    std = float(np.std(X_train))

    if std < 1e-8:
        std = 1.0

    return mean, std


def apply_global_standardizer(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


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

    Corresponding merged global positions:
        random     = [0:10000]
        m-gather   = [10000:20000]

        val        = [8000:9000] + [18000:19000]
        test       = [9000:10000] + [19000:20000]
    """
    assert random_DS.shape[0] >= 10000
    assert mgather_DS.shape[0] >= 10000

    random_DS = random_DS[:10000]
    mgather_DS = mgather_DS[:10000]

    # labels
    y_random = np.zeros(10000, dtype=np.float32)
    y_mgather = np.ones(10000, dtype=np.float32)

    # train
    X_train = np.concatenate([
        random_DS[0:8000],
        mgather_DS[0:8000]
    ], axis=0)

    y_train = np.concatenate([
        y_random[0:8000],
        y_mgather[0:8000]
    ], axis=0)

    # val
    X_val = np.concatenate([
        random_DS[8000:9000],
        mgather_DS[8000:9000]
    ], axis=0)

    y_val = np.concatenate([
        y_random[8000:9000],
        y_mgather[8000:9000]
    ], axis=0)

    # test
    X_test = np.concatenate([
        random_DS[9000:10000],
        mgather_DS[9000:10000]
    ], axis=0)

    y_test = np.concatenate([
        y_random[9000:10000],
        y_mgather[9000:10000]
    ], axis=0)

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
# Model 1: Residual TCN / ModernTCN-like
# =========================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1, dropout=0.1):
        super().__init__()

        assert kernel_size % 2 == 1, "kernel_size should be odd."

        padding = dilation * (kernel_size // 2)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.act(out)

        return out


class ResTCNBinaryClassifier(nn.Module):
    """
    Recommended main model:
    Input:  [B, 6161, 2]
    Output: [B, 1] logits
    """
    def __init__(self, in_channels=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=33, padding=16),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        self.blocks = nn.Sequential(
            ResidualBlock1D(64, 64, kernel_size=7, dilation=1),
            ResidualBlock1D(64, 64, kernel_size=7, dilation=2),
            ResidualBlock1D(64, 128, kernel_size=7, dilation=4),
            ResidualBlock1D(128, 128, kernel_size=7, dilation=8),
            ResidualBlock1D(128, 128, kernel_size=7, dilation=16),
            ResidualBlock1D(128, 128, kernel_size=7, dilation=32),
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
def get_model(model_name="res_tcn"):
    """
    model_name:
        "res_tcn"    -> preferred
        "inception"  -> alternative
    """
    if model_name == "res_tcn":
        model = ResTCNBinaryClassifier(in_channels=2)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model.to(device)


# =========================
# Save / Load model
# =========================
def save_model(model, filename, model_name):
    output_dir = os.path.dirname(filename)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name
    }, filename)

    print(f"Model saved to {filename}")


def load_model(filename):
    checkpoint = torch.load(filename, map_location=device)
    model_name = checkpoint.get('model_name', 'res_tcn')

    model = get_model(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded from {filename}, model_name = {model_name}")
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

    # Choose model:
    # "res_tcn"   : preferred, Residual TCN / ModernTCN-like
    # "inception" : alternative, InceptionTime-like
    model_name = "res_tcn"

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
    random_file = r'1_random_test_trace_5100.csv'
    mgather_file = r'm_1_test_trace_5100.csv'

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
    # Case B: If you have a single combined file, replace Case A with the block below
    # =========================
    """
    all_file = r'all_trace.csv'

    all_DS = read_ds_file(
        how_many_to_read=20000,
        filename=all_file,
        alternating_key_ds=alternating_key_ds
    )

    all_DS = crop_to_6161(all_DS)

    X_train, y_train, X_val, y_val, X_test, y_test = split_concat_ordered_data(all_DS)
    """

    # =========================
    # Standardization
    # =========================
    mean, std = fit_global_standardizer(X_train)

    X_train = apply_global_standardizer(X_train, mean, std)
    X_val = apply_global_standardizer(X_val, mean, std)
    X_test = apply_global_standardizer(X_test, mean, std)

    print(f"\nNormalizer: mean={mean:.6f}, std={std:.6f}")

    # =========================
    # Build DataLoader
    # =========================
    train_dataset = SpectrumBinaryDataset(X_train, y_train, use_dual_channel=False)
    val_dataset = SpectrumBinaryDataset(X_val, y_val, use_dual_channel=False)
    test_dataset = SpectrumBinaryDataset(X_test, y_test, use_dual_channel=False)

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
        model_filename = r'model\binary\m_5100_1.pth'
        model = load_model(model_filename)
    else:
        model = get_model(model_name)

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
            f'my_binary_{model_name}_{timestamp}.pth'
        )

        save_model(model, model_filename, model_name)

    # =========================
    # Test set evaluation
    # =========================
    print("\nEvaluate on test set...")
    test_metrics = evaluate_binary(model, test_loader, threshold=0.5)

    print("\nTest set results:")
    print(f"Loss      = {test_metrics['loss']:.6f}")
    print(f"Accuracy  = {test_metrics['acc']:.4f}")
    print(f"Precision = {test_metrics['precision']:.4f}")
    print(f"Recall    = {test_metrics['recall']:.4f}")
    print(f"F1        = {test_metrics['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(f"TP = {test_metrics['tp']}   FP = {test_metrics['fp']}")
    print(f"FN = {test_metrics['fn']}   TN = {test_metrics['tn']}")

    # =========================
    # Save prediction probabilities
    # =========================
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    pred_output_filename = (
        f'model/binary/'
        f'test_predictions_{model_name}_{timestamp}.csv'
    )

    save_binary_predictions(
        probs=test_metrics["probs"],
        labels=test_metrics["labels"],
        output_filename=pred_output_filename
    )

    
    print("loss =", test_metrics["loss"])
    print("acc =", test_metrics["acc"])
    print("precision =", test_metrics["precision"])
    print("recall =", test_metrics["recall"])
    print("f1 =", test_metrics["f1"])
    print("FP =", test_metrics["fp"])
    print("FN =", test_metrics["fn"])

    probs = test_metrics["probs"]
    labels = test_metrics["labels"]

    print("prob random mean =", probs[labels == 0].mean())
    print("prob random max  =", probs[labels == 0].max())
    print("prob weak mean   =", probs[labels == 1].mean())
    print("prob weak min    =", probs[labels == 1].min())



if __name__ == "__main__":
    main()