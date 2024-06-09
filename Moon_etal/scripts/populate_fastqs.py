import os
import subprocess
import pandas as pd
import gzip
import shutil
import argparse

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_and_gzip(sra_run_id, mixture, output_path):
    fastq_path = f"{output_path}/{sra_run_id}.fastq"
    gzipped_path = f"{output_path}/{mixture}.fastq.gz"

    if os.path.exists(gzipped_path):
        print(f"File {gzipped_path} already exists. Skipping download and compression.")
        return

    if not os.path.exists(fastq_path):
        print(f"Downloading {sra_run_id} to {fastq_path}")
        subprocess.run(['fasterq-dump', sra_run_id, '--stdout'], stdout=open(fastq_path, 'w'))
    else:
        print(f"Found existing file {fastq_path}. Skipping download.")

    print(f"Compressing {fastq_path} to {gzipped_path}")
    with open(fastq_path, 'rb') as f_in, gzip.open(gzipped_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    os.remove(fastq_path)

def process_data(file_path, base_directory):
    df = pd.read_csv(file_path, sep='\t')

    for _, row in df.iterrows():
        sample_name = row['Sample name']
        sra_run_id = row['SRA Run ID']
        mixture = row['Mixture']

        output_directory = os.path.join(base_directory, f"MixedControl-{sample_name}-fastqs/output/porechop_kraken_trimmed")
        create_directory(output_directory)

        print(f"Processing {sra_run_id} from lane {sample_name}")
        download_and_gzip(sra_run_id, mixture, output_directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and process SRA data for a given table.")
    parser.add_argument(
        "--table_path", 
        type=str, 
        default="metadata/PRJNA1031245_sra_data.tsv", 
        help="Path to the TSV file containing the SRA data (default: metadata/PRJNA1031245_sra_data.tsv)"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="data", 
        help="Base directory for output files (default: data)"
    )
    args = parser.parse_args()

    process_data(args.table_path, args.out)