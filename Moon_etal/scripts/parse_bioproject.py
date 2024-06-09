import sys
import argparse
import requests
import xmltodict
import pandas as pd
import time
import random
import re

# NOTE: Random short delays are added to space out requests

def extract_from_xml(xml_string, field_name):
    try:
        parsed = xmltodict.parse(xml_string)
        return parsed.get(field_name, None)
    except Exception as e:
        print(f"Error parsing XML for field {field_name}: {e}")
        return None

# Function to manually extract specified element from malformed XML
def extract_xml_element(exp_xml, element):
    pattern = rf'<{element}>([^<]+)</{element}>'
    match = re.search(pattern, exp_xml)
    if match:
        return match.group(1)
    return None

def get_biosample_info(biosample_id):
    time.sleep(1)
    if biosample_id.startswith("SAMN"):
        biosample_id = biosample_id[4:]
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=biosample&id={biosample_id}"
    for attempt in range(5):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = xmltodict.parse(response.content)
                if 'eSummaryResult' in data and 'DocumentSummarySet' in data['eSummaryResult'] and 'DocumentSummary' in data['eSummaryResult']['DocumentSummarySet']:
                    document_summary = data['eSummaryResult']['DocumentSummarySet']['DocumentSummary']
                    if isinstance(document_summary, list):
                        document_summary = document_summary[0]
                    
                    # Extract title-based sample name
                    title = document_summary.get('Title', '')
                    sample_name = title.split(' ')[0] if title else None
                    
                    # Extract attribute-based sample name
                    sample_data = document_summary.get('SampleData', '')
                    if isinstance(sample_data, str):
                        sample_data = xmltodict.parse(sample_data)
                    attributes = sample_data.get('BioSample', {}).get('Attributes', {}).get('Attribute', [])
                    mixture_name = None
                    if isinstance(attributes, list):
                        for attribute in attributes:
                            if attribute.get('@attribute_name') == 'sample_name':
                                mixture_name = attribute.get('#text')
                                break
                    elif isinstance(attributes, dict):
                        if attributes.get('@attribute_name') == 'sample_name':
                            mixture_name = attributes.get('#text')
                    
                    return sample_name, mixture_name

                else:
                    print(f"Unexpected response structure: {data}")
            else:
                raise Exception(f"HTTP error: {response.status_code}")
        except (Exception, ConnectionError) as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep((2 ** attempt) + random.random())
    return None, None

def get_bioproject_data(bioproject_id, retmax=1000):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=sra&term={bioproject_id}[BioProject]&retmax={retmax}"
    response = requests.get(url)
    data = xmltodict.parse(response.content)
    ids = data['eSearchResult']['IdList']['Id']

    records = []

    for sra_id in ids:
        print(f"Processing {sra_id}")
        time.sleep(1)
        sra_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=sra&id={sra_id}"
        for attempt in range(5):
            try:
                sra_response = requests.get(sra_url)
                if sra_response.status_code == 200:
                    sra_data = xmltodict.parse(sra_response.content)
                    sra_summary = sra_data['eSummaryResult']['DocSum']
                    
                    sra_entry = {
                        'SRA ID': sra_summary['Id'],
                        'SRA Run ID': None,
                        'BioSample ID': None,
                        'Sample name': None,
                        '# of Bases': None,
                        'Size': None,
                        'Published': None,
                        'Layout': None,
                        'Link': None,
                        'Mixture': None,
                    }

                    for item in sra_summary['Item']:
                        name = item['@Name']
                        if name == 'Runs':
                            run_xml = item['#text']
                            run_info = extract_from_xml(run_xml, 'Run')
                            if run_info:
                                sra_entry['SRA Run ID'] = run_info.get('@acc')
                                sra_entry['# of Bases'] = run_info.get('@total_bases')
                                sra_entry['Size'] = run_info.get('@total_size')
                        elif name == 'ExpXml':
                            exp_xml = item['#text']
                            biosample_id = extract_xml_element(exp_xml, "Biosample")

                            if biosample_id:
                                sra_entry['BioSample ID'] = biosample_id
                                title, mixture = get_biosample_info(biosample_id)
                                sra_entry['Sample name'] = title
                                sra_entry['Mixture'] = mixture
                            layout = extract_xml_element(exp_xml, "LIBRARY_LAYOUT")
                            sra_entry['Layout'] = layout
                        elif name == 'CreateDate':
                            sra_entry['Published'] = item['#text']
                    
                    if sra_entry['SRA Run ID']:
                        sra_entry['Link'] = f"https://trace.ncbi.nlm.nih.gov/Traces/sra/?run={sra_entry['SRA Run ID']}"
                    
                    records.append(sra_entry)
                    break
                else:
                    raise Exception(f"HTTP error: {sra_response.status_code}")
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                time.sleep((2 ** attempt) + random.random())
    return records

def save_to_files(records, bioproject_id):
    df = pd.DataFrame(records)
    df.to_csv(f"{bioproject_id}_sra_data.tsv", sep='\t', index=False)
    df.to_excel(f"{bioproject_id}_sra_data.xlsx", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and save SRA data for a given BioProject.")
    parser.add_argument(
        "--bioproject_id", 
        type=str, 
        default="PRJNA1031245", 
        help="BioProject ID to fetch data for (default: PRJNA1031245)"
    )
    parser.add_argument(
        "--retmax", 
        type=int, 
        default=1000, 
        help="Maximum number of records to fetch (default: 1000)"
    )
    args = parser.parse_args()

    bioproject_id = args.bioproject_id
    retmax = args.retmax
    records = get_bioproject_data(bioproject_id, retmax)
    save_to_files(records, bioproject_id)
    print(f"Data for BioProject {bioproject_id} has been saved to TSV and Excel files.")