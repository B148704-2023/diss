# Moon et al. 2024

Contains data and files from:

Moon JF, Kunkleman S, Taylor W, Harris A, Gibbs C, Schlueter J. 2024. A Gold Standard Dataset for Lineage Abundance Estimation from Wastewater. medRxiv 2024.02.15.24302811; doi:<https://doi.org/10.1101/2024.02.15.24302811>

Data: NCBI BioProject PRJNA1031245

Code and metadata: <https://github.com/enviro-lab/benchmark-deconvolute>

Paper: <https://www.medrxiv.org/content/10.1101/2024.02.15.24302811v1.full#ref-46>

## Preparing the dataset

These are the steps necessary to download and process the dataset

## 0. Files present in the folder

`./MixedControl-{lane}-fastqs/`: Base folder for each lane from Moon et al., containing a csv with some additional sample information. Copied from the [Moon et al. data repository](https://github.com/enviro-lab/benchmark-deconvolute/tree/main/ont)
`./metadata/expected_abundances/`: Folder with the expected abundance data for each lane, and also a list of mixtures dropped from the analysis. Copied from [Moon et al. data repository](https://github.com/enviro-lab/benchmark-deconvolute/tree/main/expected_abundances)

## 1. parse_bioproject.py 

Parses the given BioProject and creates a table of samples and associated metadata

```
./parse_bioproject.py -h
usage: parse_bioproject.py [-h] [--bioproject_id BIOPROJECT_ID] [--retmax RETMAX]

Fetch and save SRA data for a given BioProject.

options:
  -h, --help            show this help message and exit
  --bioproject_id BIOPROJECT_ID
                        BioProject ID to fetch data for (default: PRJNA1031245)
  --retmax RETMAX       Maximum number of records to fetch (default: 1000)
```

Output table present in this directory within `./metadata/`

### 2. fetch_and_format_srr.py

TBD

### 3. covid-analysis pipeline 

TBD 
<https://github.com/enviro-lab/covid-analysis> 

