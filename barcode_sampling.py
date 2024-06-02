import sys
import random
import argparse
import pandas as pd
import numpy as np


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    else:
        np.random.seed(None)

    # Read barcodes file
    try:
        bcs = pd.read_csv(args.barcodes)
        print(f"Barcode dataset loaded from: {args.barcodes}")
    except FileNotFoundError:
        print(f"Error: The file {args.barcodes} does not exist.")
        sys.exit(1)

    for i in range(args.replicates):
        # If not fixed num_lineages, sample num_lineages from uniform dist
        if not args.num_lineages:
            n = np.random.randint(args.min_lineages, args.max_lineages + 1)
        else:
            n = args.num_lineages

        # Sample lineages and assign frequencies
        lin_freqs = sample_lineages(bcs, n, args.frequency_distribution,
                                    args.alpha)

        # Compute SNV frequencies
        snv_freqs = compute_snv_frequencies(bcs, lin_freqs)

        # Display results (for now)
        print(snv_freqs)


def compute_snv_frequencies(bcs, lineage_freqs, filter=True):
    """
    Computes the SNV frequencies based on the sampled lineage frequencies.

    Parameters:
    bcs (pd.DataFrame): The barcodes dataframe.
    lineage_freqs (dict): A dictionary containing sampled lineage names and
                          their frequencies.
    filter (bool): Whether to filter out positions where the sum of ALT
                   frequencies is 0. Default is True.

    Returns:
    dict: A dictionary containing SNV positions and their frequencies.
    """
    snv_freqs = {}

    for lineage, freq in lineage_freqs.items():
        lineage_snv = bcs[bcs.iloc[:, 0] == lineage].iloc[:, 1:].to_dict(
            orient='records')[0]
        for snv, snv_freq in lineage_snv.items():
            pos = int(snv[1:-1])
            ref = snv[0]
            alt = snv[-1]
            if pos not in snv_freqs:
                snv_freqs[pos] = {'REF': ref, 'ALT': {}}
            else:
                if snv_freqs[pos]['REF'] != ref:
                    print(f"Warning: Inconsistent REF state at position {pos}"
                          f"Expected {snv_freqs[pos]['REF']}, found {ref}")

            if alt not in snv_freqs[pos]['ALT']:
                snv_freqs[pos]['ALT'][alt] = 0
            snv_freqs[pos]['ALT'][alt] += snv_freq * freq

    # if filter is True, remove sites where only state is ref
    if filter:
        snv_freqs = {
            pos: data
            for pos, data in snv_freqs.items()
            if sum(data['ALT'].values()) > 0
        }
    # Sort the dictionary by position
    snv_freqs = dict(sorted(snv_freqs.items()))

    return snv_freqs


def sample_lineages(bcs, num_lineages, frequency_distribution, alpha):
    """
    Samples lineages from the given barcodes dataframe and assigns frequencies
    based on the specified distribution type.

    Parameters:
    bcs (pd.DataFrame): The barcodes dataframe.
    num_lineages (int): The number of lineages to sample.
    frequency_distribution (str): The type of frequency distribution to use
                                  ('uniform' or 'dirichlet').
    alpha (float): The alpha parameter for the Dirichlet distribution
                   when 'dirichlet' is chosen.

    Returns:
    dict: A dictionary containing the sampled lineage names and the assigned
          frequencies.

    Raises:
    ValueError: If an invalid frequency distribution type is provided.
    """
    sampled_bcs = bcs.sample(num_lineages)

    if frequency_distribution == 'uniform':
        frequencies = np.ones(num_lineages) / num_lineages  # Balanced
    elif frequency_distribution == 'dirichlet':
        alpha = np.linspace(alpha, 1, num_lineages)  # Skewed
        frequencies = np.random.dirichlet(alpha)
    else:
        raise ValueError(
            f"Invalid frequency distribution type: {frequency_distribution}")

    freqs = dict(zip(sampled_bcs.iloc[:, 0], frequencies))

    return freqs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample from Freyja barcode database file")

    parser.add_argument(
        '-s', '--seed', type=int,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '-b', '--barcodes', type=str, required=True,
        help='Freyja barcodes database file'
    )
    parser.add_argument(
        '-n', '--num_lineages', type=int, required=False,
        help='Fixed number of lineages to sample'
    )
    parser.add_argument(
        '-m', '--min_lineages', type=int, default=2,
        help='Minimum number of lineages to sample (for range sampling)'
    )
    parser.add_argument(
        '-M', '--max_lineages', type=int, default=10,
        help='Maximum number of lineages to sample (for range sampling)'
    )
    parser.add_argument(
        '-f', '--frequency_distribution', choices=['uniform', 'skewed'],
        default='uniform',
        help='Distribution of frequencies assigned to sampled lineages'
    )
    parser.add_argument(
        '-a', '--alpha', type=float, default=2.0,
        help='Alpha parameter for Dirichlet distribution'
    )
    parser.add_argument(
        '-r', '--replicates', type=int, default=1,
        help='Number of replicates for sampling'
    )

    args = parser.parse_args()
    main(args)