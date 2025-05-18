import os
import random
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence


def collate_concept(batch, device="cpu"):
    # Each item in batch, after BaseMetaDataset.__getitem__ for MetaBitConceptsDataset, is:
    # item[0]: X_s_processed_tensor (from original task_data[0])
    # item[1]: y_s_tensor (from original task_data[2])
    # item[2]: X_q_processed_tensor (from original task_data[3])
    # item[3]: y_q_tensor (from original task_data[5])

    X_s_list = [item[0].to(device) for item in batch]
    y_s_list = [item[1].to(device) for item in batch]
    X_q_list = [item[2].to(device) for item in batch]
    y_q_list = [item[3].to(device) for item in batch]

    # Pad sequences: batch_first=True makes the output (batch_size, max_len, num_features)
    # For labels, padding_value=0.0 is used. This might be okay if 0 is a neutral/non-class for BCE.
    # If labels are strictly 0 or 1, padding with 0 means these will be treated as class 0 instances.
    X_s_padded = pad_sequence(X_s_list, batch_first=True, padding_value=0.0)
    y_s_padded = pad_sequence(y_s_list, batch_first=True, padding_value=0.0)
    X_q_padded = pad_sequence(X_q_list, batch_first=True, padding_value=0.0)
    y_q_padded = pad_sequence(y_q_list, batch_first=True, padding_value=0.0)

    return X_s_padded, y_s_padded, X_q_padded, y_q_padded


def collate_default(batch, device="cpu"):
    def move_to_device(data):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, list):
            return [move_to_device(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(move_to_device(item) for item in data)
        elif isinstance(data, dict):
            return {key: move_to_device(value) for key, value in data.items()}
        else:
            return data

    batch = default_collate(batch)
    return move_to_device(batch)


def get_collate(experiment: str, device="cpu"):
    if experiment in ["concept", "mod"]:
        return lambda batch: collate_concept(batch, device=device)
    else:
        return lambda batch: collate_default(batch, device=device)


def set_random_seeds(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 1 gpu


def save_model(meta, save_dir="state_dicts", file_prefix="meta_learning"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(meta.state_dict(), f"{save_dir}/{file_prefix}.pth")
    print(f"Model saved to {save_dir}/{file_prefix}.pth")


def calculate_accuracy(predictions, targets):
    if predictions.shape[1] > 1:
        predictions = predictions.argmax(dim=1)
    else:
        predictions = (predictions > 0.0).float()
    correct = (predictions == targets).sum().item()
    accuracy = correct / targets.numel()
    return accuracy
