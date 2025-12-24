#!/usr/bin/env python 
import csv
import argparse

# NOTE: This script averages retention times for metabolites that appear multiple times
# across datasets. The output contains one copy of each unique metabolite.

def main():
    parser = argparse.ArgumentParser(
        description='Process dataset CSV to create averaged input for graphormer-RT.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
This script processes the output from initiate_dataset.py and creates input for graphormer-RT.

For metabolites that appear more than once (unknown metabolites appearing across multiple datasets),
this script averages their retention times. The output contains only one copy of each unique metabolite,
and the predictions will serve as a comparable dictionary for reference.

Output format: setup_id,smiles,averaged_retention_time

Example:
  python initiate_input_for_graphormer_eval.py --input_file_name combined_data.csv \\
    --output_file_name averaged_for_graphormer.csv --setup_id "0001"
        '''
    )
    
    parser.add_argument('--input_file_name', '-i', type=str, required=True,
                        help='Input CSV file (output from initiate_dataset.py)')
    parser.add_argument('--output_file_name', '-o', type=str, required=True,
                        help='Output CSV file name for graphormer-RT input')
    parser.add_argument('--setup_id', '-s', type=str, required=True,
                        help='Setup ID for graphormer-RT to reference')
    
    args = parser.parse_args()
    
    input_file_path = args.input_file_name
    output_file_path = args.output_file_name
    setup_id = args.setup_id
    
    averaged_smiles = {}
    
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header line
        
        for row in reader:
            file_name = row[0]        # File Name
            retention_time = float(row[2])  # Retention Time (min)
            smiles = row[3]           # smiles
            
            # Accumulate retention times and counts for averaging
            if smiles in averaged_smiles:
                averaged_smiles[smiles][0] += retention_time
                averaged_smiles[smiles][1] += 1
            else:
                averaged_smiles[smiles] = [retention_time, 1]
    
    # Write to output file
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for smiles, values in averaged_smiles.items():
            avg_rt = values[0] / values[1]  # Calculate average retention time
            outfile.write(f"{setup_id},{smiles},{avg_rt}\n")
    
    print(f"Successfully created {output_file_path} with {len(averaged_smiles)} unique metabolites")


if __name__ == '__main__':
    main()