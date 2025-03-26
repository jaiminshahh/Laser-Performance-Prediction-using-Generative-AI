import numpy as np
import torch

def calculate_contrast(image):
    # Replace this with your chosen method of calculating contrast
    # Using np.nanstd to ignore NaN values
    return np.nanstd(image)

def calculate_mape(y_true, y_pred):
    # Avoid division by zero
    non_zero_mask = y_true != 0
    return torch.mean((torch.abs(y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]) * 100).item()
