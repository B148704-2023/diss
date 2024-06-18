import pandas as pd
import numpy as np
import random
import time

def generate_training_examples(barcodes, n, filter=True):
    """
    Generate traning examples of known lineage frequencies and 
            corresponding SNV frequencies from barcode files.

    Parameters:
    barcodes (str): The path to the barcode file
    n (int): Number of training examples to be generated

    Returns:
    known_freqs & sorted_snv_freqs: Lists denoting the known lineage
                frequencies and SNV frequencies
    """

    start_time=time.time()

    print(f'Initialising. Generating {n} training/testing examples from: {barcodes}...')

    bcs=pd.read_csv(barcodes) #load the Barcode file

    #initialise the lists
    known_freqs=[]
    sorted_snv_freqs=[]

    for i in range(n):
    
        num_lineages=random.randint(5,10) #the number of lineages being sampled
        frequency_distribution=random.choice(['uniform','dirichlet']) #balanced vs skewed sampling
    
        sampled_bcs = bcs.sample(num_lineages)

        if frequency_distribution == 'uniform':
            frequencies = np.ones(num_lineages) / num_lineages  # Balanced
        elif frequency_distribution == 'dirichlet':
            alpha = np.linspace(2, 1, num_lineages)  # Skewed
            frequencies = np.random.dirichlet(alpha)

        lineage_freqs = dict(zip(sampled_bcs.iloc[:, 0], frequencies))
        known_freqs.append(frequencies.tolist())

        snv_freqs={}

        for lineage, freq in lineage_freqs.items():
            lineage_snv = bcs[bcs.iloc[:, 0] == lineage].iloc[:, 1:].to_dict(
                orient='records')[0]
            for snv, snv_freq in lineage_snv.items():
                    pos = int(snv[1:-1])
                    if pos not in snv_freqs:
                        snv_freqs[pos] = 0
                    snv_freqs[pos] += snv_freq * freq

        #filter out empty values
        if filter:
            snv_freqs = {
                pos: freq
                for pos, freq in snv_freqs.items()
                if freq > 0
        }

        sorted_snv_freqs.append([freq for pos, freq in sorted(snv_freqs.items())])
        if i%50==0:
            print(f'Generated {i} training examples...')

    end_time=time.time()
    elapsed=end_time-start_time

    print(f'Elapsed: {elapsed/60} minutes.')

    return known_freqs, sorted_snv_freqs