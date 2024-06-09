#!/usr/bin/env bash
set -eu

bioproject=PRJNA1031245
declare -A plates=(["05-05-23-A41"]="v4.1-ctrl" ["05-16-23-A41"]="v4.1-neg" ["06-26-23-A41"]="v4.1-pos" ["05-05-23-V2"]="v2a-ctrl" ["06-16-23-V2"]="v2a-neg" ["07-12-23-V2A"]="v2a-pos")

# Using sra-tools and entrez-direct (available through conda), collect relevant fastqs for each plate
for plate in "${!plates[@]}"; do
    title="${plates[${plate}]}"
    echo "$plate: $title"
    esearch -db sra -query "$bioproject" | esummary | xtract -pattern DocumentSummary -element LIBRARY_NAME,Run@acc | grep "$title" | while read -r name srr; do
        # Use sra-tools to download associated sample from SRA
        mixture="${name%%-*}"
        output_dir="MixedControl-${plate}-fastqs/output/porechop_kraken_trimmed"
        mkdir -p "$output_dir"

        # Check if the file already exists
        if [[ -f "${output_dir}/${mixture}.fastq.gz" ]]; then
            echo "Files for $srr ($mixture) already exist, skipping download."
        else
            echo "Downloading $srr ($mixture) ($name)"
            prefetch "$srr"
            fasterq-dump --outdir "$output_dir" "${srr}/${srr}.sra"

            # Check for presence of expected files and move them to the final destination
            if [[ -f "${output_dir}/${srr}.fastq.gz" ]]; then
                mv "${output_dir}/${srr}.fastq.gz" "${output_dir}/${mixture}.fastq.gz"
            elif [[ -f "${output_dir}/${srr}.fastq" ]]; then
                echo "zipping up ${srr}.fastq"
                gzip "${output_dir}/${srr}.fastq"
                mv "${output_dir}/${srr}.fastq.gz" "${output_dir}/${mixture}.fastq.gz"
            else
                echo "Missing fastq for $srr ($mixture)"
            fi
            rm -rf "$srr"
        fi
    done
done