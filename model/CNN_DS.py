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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

num_bits = 12323
num_dist = (num_bits // 2) + 1
start_cut = 0
end_cut = 0

# Label distance range control
# Now by default keeps all positive distances 1,2,3,...
label_min_dist = 40
label_max_dist = None   # If you only want to predict 1~30, change to 30

# Custom data loader
class TraceDataset(Dataset):
    def __init__(self, DS_data, label_data, use_dual_channel=False):
        self.DS_data = np.asarray(DS_data, dtype=np.float32)
        self.label_data = np.asarray(label_data, dtype=np.float32)
        self.use_dual_channel = use_dual_channel

    def __len__(self):
        return len(self.DS_data)

    @staticmethod
    def smooth_spectrum(x):
        """
        Light smoothing: 3-point weighted average
        """
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        x_pad = np.pad(x, (1, 1), mode='edge')
        y = np.convolve(x_pad, kernel, mode='valid')
        return y.astype(np.float32)

    def __getitem__(self, idx):
        x = self.DS_data[idx]          # (seq_len,)
        y = self.label_data[idx]       # (seq_len,)

        if self.use_dual_channel:
            x_smooth = self.smooth_spectrum(x)
            x = np.stack([x, x_smooth], axis=-1)   # (seq_len, 2)
        else:
            x = x[:, np.newaxis]                  # (seq_len, 1)

        y = y[:, np.newaxis]                      # (seq_len, 1)

        return torch.from_numpy(x), torch.from_numpy(y)

def top_20_percent_lowering_peaks(y_true, y_pred):
    """
    Compute the proportion of lowering peaks in the top 20% predictions:
    among the top 20% positions with highest predicted values, 
    how many ground-truth label=1 points are hit.
    """
    # Flatten tensors
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    # Get count of top 20%
    top_20_percent_count = int(round(0.2 * y_pred.numel())) 
    # y_pred.numel() returns total number of elements in the prediction tensor.
    # 0.2 * y_pred.numel() represents the count of top 20%.

    # Get indices of top 20% predictions
    _, top_indices = torch.topk(y_pred, top_20_percent_count) 
    # torch.topk finds the top top_20_percent_count elements with the largest predicted values.
    # top_indices are the position indices corresponding to these maximum values.

    # Get the ground-truth labels at these indices
    top_true_labels = y_true[top_indices]

    # Count actual lowering peaks in top 20%: count how many positions have ground-truth label = 1.
    lowering_peak_hits = torch.sum((top_true_labels == 1).float())

    # Compute proportion
    total_lowering_peaks = torch.sum((y_true == 1).float())
    proportion = lowering_peak_hits / (total_lowering_peaks)
    
    return proportion.item()

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock1D, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + identity
        out = self.relu(out)
        return out


class Conv1DModel(nn.Module):
    def __init__(self):
        super(Conv1DModel, self).__init__()
        input_size = num_dist - (start_cut + end_cut)

        # Note: input channels changed to 2 (original spectrum + smoothed spectrum)
        # The initial kernel_size was changed from 34 to 33 to more stably preserve sequence length
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=33, padding=16),
            nn.ReLU()
        )

        self.res_block1 = ResidualBlock1D(32, 32, kernel_size=3)
        self.res_block2 = ResidualBlock1D(32, 64, kernel_size=3)
        self.res_block3 = ResidualBlock1D(64, 64, kernel_size=3)

        self.head = nn.Conv1d(64, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, 2)
        x = x.transpose(1, 2)  # -> (batch_size, 2, seq_len)

        x = self.stem(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.head(x)

        x = self.sigmoid(x)
        x = x.transpose(1, 2)  # -> (batch_size, seq_len, 1)
        return x

def get_model():
    """Create and return the model"""
    model = Conv1DModel().to(device)
    return model

def read_trace_file(how_many_to_read, filename):
    """Read trace file"""
    keys = []
    DS = []
    still_todo = how_many_to_read
    
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i % 2 == 0:
                # Even rows: list of integers (key)
                values = [int(value.strip()) for value in row] 
                # Even rows store the key
                # Strip spaces and convert each value to integer
                keys.append(values)
            else:
                # Odd rows: list of floats (distance spectrum)
                values = [float(value.strip()) for value in row] 
                # Odd rows store spectrum data, convert to floats
                DS.append(values)
                still_todo -= 1
            
            if still_todo == 0:
                break
    
    print(f"read {how_many_to_read - still_todo} traces")
    
    keys_array = np.array(keys)
    DS_array = np.array(DS)
    
    return keys_array, DS_array

def det_dist_in_spec(s, i):
    """Compute the distance between two positions (considering circular shift)"""
    a, b = (s, i) if s < i else (i, s)
    max_dist = num_bits // 2
    
    if (b - a) > max_dist:
        return num_bits - (b - a)
    else:
        return (b - a)

def find_middle_position(wlist):
    """Find the middle position of a contiguous block (length≥30)"""
    max_start, max_len = 0, 0
    current_start, current_len = 0, 1
    
    for i in range(1, len(wlist)):
        if wlist[i] == wlist[i-1] + 1: 
            # If current element is 1 greater than previous, still in contiguous block, 
            # increment block length
            current_len += 1
        else:
            if current_len >= max_len: 
                # If not contiguous, current block ends; 
                # if current block is longer than the known longest, update the record;
                # then start a new block from current position
                max_len = current_len
                max_start = current_start
            current_start = i
            current_len = 1

    # After the loop, also check the last segment, since it won't enter the else branch.
    if current_len >= max_len:
        max_len = current_len
        max_start = current_start
    
    if max_len < 30:
        raise ValueError("No valid block in key")
    
    middle_index = max_start + max_len // 2
    return wlist[middle_index]

def get_train_labels_from_key(wlist):
    """
    Generate training labels from key
    No longer restricts dist >= 40; earlier distances are allowed into labels.
    By default, all positive distances dist >= 1 are kept.
    If you later want to keep only 1~30, set label_max_dist = 30
    """
    interesting_points = np.zeros(num_dist, dtype=np.float32)
    blockcenter = find_middle_position(wlist)

    for pos in wlist:
        dist = det_dist_in_spec(blockcenter, pos)

        # Exclude dist=0, keep dist>=1
        if dist < label_min_dist:
            continue
        if label_max_dist is not None and dist > label_max_dist:
            continue

        interesting_points[dist] = 1.0

    return interesting_points

def get_traces(trace_to_do, filename):
    """Get trace data, keys and labels"""
    keys, DS = read_trace_file(trace_to_do, filename)

    labels = []
    for k in keys:
        labels.append(get_train_labels_from_key(k))
    labels = np.array(labels)

    print(f"First key: {keys[0]}")
    return keys, DS, labels

def normalize_data(data):
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

def print_Top_K_Evaluation(eval_pred, eval_truth, top_pred):
    """Top-K evaluation, returns the number of missed ground-truth points."""
    # This function evaluates how many ground-truth label=1 positions are 
    # covered among the top K predicted positions.
    num_top_predictions = top_pred
    top_k_indices = np.argsort(eval_pred)[-num_top_predictions:]
    # np.argsort(eval_pred) returns indices sorted from smallest to largest
    lowering_indices = np.where(eval_truth == 1)[0]
    # Find all positions where ground-truth label is 1.

    correctly_predicted_mask = np.isin(lowering_indices, top_k_indices)
    # Determine whether true positive positions are included in the top K predicted positions
    correctly_predicted_count = np.sum(correctly_predicted_mask)
    # Count the number of hits

    print(f"For {num_top_predictions} selected distances: {correctly_predicted_count} " 
          # How many ground-truth points are hit among the top K predictions
          f"out of {len(lowering_indices)} are included. "
          f"Missing: {len(lowering_indices) - correctly_predicted_count}") 
    # How many are still missed
    
    return len(lowering_indices) - correctly_predicted_count

def get_top_k_distances(eval_pred, top_pred):
    """
    Return the top top_pred distances with the highest predicted values.
    Results are sorted by prediction score from high to low.
    Note: if start_cut != 0, actual distance = local index + start_cut.
    """
    top_k_indices = np.argsort(eval_pred)[-top_pred:]

    # Sort by prediction value from high to low
    top_k_indices = top_k_indices[np.argsort(eval_pred[top_k_indices])[::-1]]

    # Convert back to real distance
    top_k_distances = [int(idx + start_cut) for idx in top_k_indices]
    return top_k_distances


def save_strict_hit_records_to_csv(records, output_filename):
    """
    CSV format:
    Row 0: positions of test key 1 that satisfy the condition
    Row 1: the top-percent distance set predicted from that key
    Row 2: positions of test key 2 that satisfy the condition
    Row 3: the top-percent distance set predicted from that key
    ...

    No extra info such as sample_index / top_percent / missed is written.
    """
    output_dir = os.path.dirname(output_filename)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    with open(output_filename, "w", newline="") as f:
        writer = csv.writer(f)

        for rec in records:
            key = rec["key"]
            distances = rec["top_distances"]

            writer.writerow(key)
            writer.writerow(distances)

    print(f"Saved {len(records)} strict-hit records to: {output_filename}")

def save_model(model, filename):
    """Save model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__
    }, filename)
    print(f"Model saved to {filename}")

def load_model(filename, model_class=None):
    """Load model"""
    checkpoint = torch.load(filename, map_location=device)
    
    if model_class is None:
        # If no model class is provided, create a default model
        model = Conv1DModel()
    else:
        model = model_class()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {filename}")
    return model

def train_model(model, train_loader, val_loader=None, epochs=40, patience=6, min_delta=1e-4):
    """
    Use validation set for early stopping:
    - Monitor val_loss
    - Stop if no improvement for consecutive patience epochs
    - Restore best parameters after training ends
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_losses = []
    val_metrics = []

    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        # =========================
        # Training phase
        # =========================
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        avg_train_loss = epoch_loss / max(batch_count, 1)
        train_losses.append(avg_train_loss)

        # =========================
        # Validation phase
        # =========================
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_metric = 0.0
            val_batches = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                    loss = criterion(output, target)
                    val_loss += loss.item()

                    metric_val = top_20_percent_lowering_peaks(target, output)
                    val_metric += metric_val
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            avg_val_metric = val_metric / max(val_batches, 1)

            val_losses.append(avg_val_loss)
            val_metrics.append(avg_val_metric)

            print(f'Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, '
                  f'Val Loss={avg_val_loss:.4f}, Val Metric={avg_val_metric:.4f}')

            # Early stopping logic
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print("Validation loss improved. Save current best model in memory.")
            else:
                patience_counter += 1
                print(f'No improvement. EarlyStop patience: {patience_counter}/{patience}')

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        else:
            print(f'Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}')

    # Restore best parameters
    model.load_state_dict(best_model_state)
    print(f"Restored best model with val_loss = {best_val_loss:.4f}")

    return train_losses, val_losses, val_metrics

def load_split_data_from_blocks(block_list, file_template):
    """
    Each block has 10000 samples:
    - 0:8000     -> train
    - 8000:9000  -> val
    - 9000:10000 -> test
    Final concatenation:
    - train: 40000
    - val:   5000
    - test:  5000

    Additionally returns:
    - keys_test: keys corresponding to the test set
    """
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    keys_test_list = []

    for block_id in block_list:
        filename = file_template.format(block_id)
        print(f"\nLoading block {block_id} from {filename}")

        keys, DS, labels = get_traces(10000, filename)
        DS = normalize_data(DS)

        DS = DS[:, (0 + start_cut):(num_dist - end_cut)].astype(np.float32)
        labels = labels[:, (0 + start_cut):(num_dist - end_cut)].astype(np.float32)

        X_train_list.append(DS[0:8000])
        y_train_list.append(labels[0:8000])

        X_val_list.append(DS[8000:9000])
        y_val_list.append(labels[8000:9000])

        X_test_list.append(DS[9000:10000])
        y_test_list.append(labels[9000:10000])

        keys_test_list.append(keys[9000:10000])

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    keys_test = np.concatenate(keys_test_list, axis=0)

    print("\nSplit done:")
    print(f"Train: {X_train.shape[0]}")
    print(f"Val:   {X_val.shape[0]}")
    print(f"Test:  {X_test.shape[0]}")
    print(f"Test keys: {keys_test.shape[0]}")

    return X_train, y_train, X_val, y_val, X_test, y_test, keys_test

def main(function=0):
    """
    function=0: original CNN_DS evaluation mode
    function=1: generate D-set mode (export strict-hit keys with predicted distances)
    """
    # =========================
    # Parameter configuration
    # =========================
    file_template = r'\test_trace_b{}.csv'
    block_list = [30, 31, 32, 33, 34]

    # Note: model architecture has been changed (dual channel + residual)
    # So the old single-channel model cannot be loaded directly
    use_old_model = 0

    train_batch_size = 32
    val_batch_size = 16
    test_batch_size = 16

    epochs = 40
    patience = 6

    # =========================
    # Read and split data
    # =========================
    X_train, y_train, X_val, y_val, X_test, y_test, keys_test = load_split_data_from_blocks(
        block_list=block_list,
        file_template=file_template
    )

    # =========================
    # Build DataLoader
    # =========================
    train_dataset = TraceDataset(X_train, y_train, use_dual_channel=False)
    val_dataset = TraceDataset(X_val, y_val, use_dual_channel=False)
    test_dataset = TraceDataset(X_test, y_test, use_dual_channel=False)

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
        model_filename = r'model\binary\xxx.pth'
        model = load_model(model_filename, Conv1DModel)
    else:
        model = get_model()

        print("\nStart training model...")
        train_losses, val_losses, val_metrics = train_model(
            model,
            train_loader,
            val_loader=val_loader,
            epochs=epochs,
            patience=patience
        )

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_filename = f'model\binary\xxx.pth'
        save_model(model, model_filename)

    # =========================
    # Test set evaluation
    # =========================
    print("\nRunning predictions on test set...")
    model.eval()

    prediction_batches = []
    ground_truth_batches = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            output = model(data)

            prediction_batches.append(output.cpu().numpy())
            ground_truth_batches.append(target.numpy())

    predictions_np = np.concatenate(prediction_batches, axis=0)
    y_test_np = np.concatenate(ground_truth_batches, axis=0)

    num_test_samples = predictions_np.shape[0]
    effective_num_dist = num_dist - (start_cut + end_cut)

    reshaped_predictions = predictions_np.reshape(num_test_samples, effective_num_dist)
    reshaped_ground_truth = y_test_np.reshape(num_test_samples, effective_num_dist)

    top_strict = [0, 0, 0, 0]   # missed <= 0
    top_loose  = [0, 0, 0, 0]   # missed <= 2
    top_percentages = [0.05, 0.10, 0.15, 0.20]

    # For function=1: export strict-hit records
    if function == 1:
        # Set to 0.15 to save Top-15%, set to 0.20 to save Top-20%
        export_percentage = 0.15
        strict_hit_records = []

    for i in range(num_test_samples):
        eval_pred = reshaped_predictions[i]
        eval_truth = reshaped_ground_truth[i]

        for j, percentage in enumerate(top_percentages):
            top_pred_count = round(effective_num_dist * percentage)

            missed = print_Top_K_Evaluation(
                eval_pred,
                eval_truth,
                top_pred_count
            )

            if missed <= 0:
                top_strict[j] += 1

                # For function=1: save strict-hit samples for the specified export_percentage
                if function == 1 and abs(percentage - export_percentage) < 1e-12:
                    top_distances = get_top_k_distances(
                        eval_pred,
                        top_pred_count
                    )

                    strict_hit_records.append({
                        "key": [int(x) for x in keys_test[i]],
                        "top_distances": top_distances
                    })

            if missed <= 2:
                top_loose[j] += 1

        print()

    print("Evaluation results:")
    for j, percentage in enumerate(top_percentages):
        strict_rate = top_strict[j] / num_test_samples if num_test_samples > 0 else 0
        loose_rate = top_loose[j] / num_test_samples if num_test_samples > 0 else 0

        print(
            f"Top {int(percentage*100)}%: "
            f"missed<=0 -> {top_strict[j]} / {num_test_samples} = {strict_rate:.4f} ; "
            f"missed<=2 -> {top_loose[j]} / {num_test_samples} = {loose_rate:.4f}"
        )

    # =========================
    # For function=1: save test keys and corresponding distances for which missed <= 0
    # =========================
    if function == 1:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        export_percent_int = int(round(export_percentage * 100))

        strict_hit_output_filename = (
            f'model/missed_0_data/'
            f'test_keys_top{export_percent_int}_missed0_{timestamp}.csv'
        )

        save_strict_hit_records_to_csv(
            strict_hit_records,
            strict_hit_output_filename
        )

if __name__ == "__main__":
    main(function=0)