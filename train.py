import pandas as pd
import numpy as np
import random
import time
import torch

import sys


# TODO: Need to add options controlling parameter space
# TODO: Can probably skip uniform as an option as this is covered by
#       alpha=1.0 in Dirichlet case
# TODO: (later) We can add Gaussian noise to the lineage_freqs to increase 
#       variation in the training set, but need to make sure these sum to 1
#       after adding noise.
def generate_training_examples(barcodes, n):
    """
    Generate training examples of known lineage frequencies and 
    corresponding SNV frequencies from barcode files.

    Parameters:
    barcodes (str): The path to the barcode file
    n (int): Number of training examples to be generated

    Returns:
    known_freqs_tensor: 2D tensor of known lineage frequencies
    snv_freqs_tensor: 2D tensor of SNV frequencies
    """

    start_time = time.time()

    print(f'Initializing. Generating {n} training/testing examples from: {barcodes}...')

    bcs = pd.read_csv(barcodes)
    snvs = bcs.iloc[:, 1:].values
    num_lineages_total = bcs.shape[0]

    # Initialize the lists
    # not dicts as the order needs to be consistent across examples
    known_freqs = []
    snv_freqs = []

    for i in range(n):
        num_lineages_sampled = random.randint(5, 10)  # The number of lineages sampled
        frequency_distribution = random.choice(['uniform', 'dirichlet'])  # Balanced vs skewed sampling

        sampled_indices = random.sample(range(num_lineages_total), num_lineages_sampled)

        if frequency_distribution == 'uniform':
            frequencies = np.ones(num_lineages_sampled) / num_lineages_sampled  # Balanced
        elif frequency_distribution == 'dirichlet':
            alpha_start = np.random.uniform(1.0, 20.0)
            alphas = np.linspace(alpha_start, 1.0, num_lineages_sampled)  # Skewed
            frequencies = np.random.dirichlet(alphas)

        # Zeros to store the lineage freqs
        lineage_freqs = np.zeros(num_lineages_total)

        # Set the frequencies for the sampled lineages
        # Result should have length(num_lineages)
        for idx, freq in zip(sampled_indices, frequencies):
            lineage_freqs[idx] = freq

        known_freqs.append(lineage_freqs)

        # test print
        # print(lineage_freqs)
        # nonzero_indices = np.nonzero(lineage_freqs)[0]
        # print(lineage_freqs[nonzero_indices])

        # Compute the SNV frequencies using matrix multiplication
        # Result should have length num_snvs
        snv_freq = np.dot(lineage_freqs, snvs)
        snv_freqs.append(snv_freq)

        # test print
        # print(snv_freq)
        # nonzero_indices = np.nonzero(snv_freq)[0]
        # print(snv_freq[nonzero_indices])
        # sys.exit()

        if i % 50 == 0:
            print(f'Generated {i} training examples...')

    end_time = time.time()
    elapsed = end_time - start_time

    print(f'Elapsed: {elapsed} seconds.')

    # convert to 2D numpy arrays and return as tensors
    known_freqs_tensor = torch.tensor(
        np.array(known_freqs), dtype=torch.float32
    )
    snv_freqs_tensor = torch.tensor(
        np.array(snv_freqs), dtype=torch.float32
    )

    return known_freqs_tensor, snv_freqs_tensor

