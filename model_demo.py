import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from train import generate_training_examples

def main():
    # Use hardware accelerator if available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    known_freqs, sorted_snv_freqs = generate_training_examples(
        'data/usher_barcodes.csv', 1000
    )

    # Move tensors to the selected device
    known_freqs = known_freqs.to(device)
    sorted_snv_freqs = sorted_snv_freqs.to(device)
    dataset = TensorDataset(known_freqs, sorted_snv_freqs)

    # Example: Print the shapes of the tensors
    print(f"Shape of known_freqs tensor: {known_freqs.shape}")
    print(f"Shape of sorted_snv_freqs tensor: {sorted_snv_freqs.shape}")

    # TODO: CLI arguments to set the split
    total_size = len(dataset)
    test_size = total_size // 10  # 10% for test
    validation_size = total_size // 10  # 10% for validation
    train_size = total_size - test_size - validation_size  # 80% train

    # Split the dataset
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [train_size, validation_size, test_size]
    )

    # verify sizes
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # DataLoaders (abstracts away some things for later)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Example: Print the first batch of the train_loader (optional)
    for batch in train_loader:
        known_freqs_batch, snv_freqs_batch = batch
        print(f"Known frequencies batch shape: {known_freqs_batch.shape}")
        print(f"SNV frequencies batch shape: {snv_freqs_batch.shape}")
        break

if __name__ == "__main__":
    main()
