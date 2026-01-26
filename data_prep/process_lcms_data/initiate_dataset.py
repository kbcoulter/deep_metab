#!/usr/bin/env python 
from pathlib import Path
import pandas as pd
import argparse

# NOTE: This script excludes data without SMILES strings. It also assumes that all data comes from the same experiment (parameters are consistent).
# Output CSV structure: File Name, Mass Feature ID, Retention Time (in minutes), smiles

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate CSV files from a dataset folder into a single CSV file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Output CSV structure:
  - File Name: Name of the source file
  - Mass Feature ID: Mass feature identifier from input files
  - Retention Time (min): Retention time in minutes
  - smiles: SMILES string representation of the molecule

Example:
  python initiate_dataset.py --dataset_folder /path/to/data --output_csv output.csv \\
    --mass_feature_id_col "Mass Feature ID" --retention_time_col "Retention Time (min)" \\
    --smiles_col "smiles"
        '''
    )
    
    parser.add_argument('--dataset_folder', type=str, required=True,
                        help='Path to the folder containing dataset CSV files')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Name of the output CSV file to create')
    parser.add_argument('--mass_feature_id_col', type=str, required=True,
                        help='Column name in input files for Mass Feature ID')
    parser.add_argument('--retention_time_col', type=str, required=True,
                        help='Column name in input files for Retention Time')
    parser.add_argument('--smiles_col', type=str, required=True,
                        help='Column name in input files for SMILES string')
    
    args = parser.parse_args()
    
    folder = Path(args.dataset_folder)
    output_csv = args.output_csv
    mass_feature_id_col = args.mass_feature_id_col
    retention_time_col = args.retention_time_col
    smiles_col = args.smiles_col
    
    if not folder.exists():
        raise ValueError(f"Dataset folder does not exist: {folder}")
    
    dfs = []
    for file in folder.iterdir():
        if file.is_file():
            print(f"Extracting {file.name}...")
            
            sample_data_frame = pd.read_csv(file)
            sample_data_frame['File Name'] = file.name
            
            # Extract only required columns and drop rows with missing values
            required_cols = ['File Name', mass_feature_id_col, retention_time_col, smiles_col]
            sample_data_frame = sample_data_frame[required_cols].dropna(how='any', axis=0)
            
            # Rename columns to standardized output format
            sample_data_frame = sample_data_frame.rename(columns={
                mass_feature_id_col: 'Mass Feature ID',
                retention_time_col: 'Retention Time (min)',
                smiles_col: 'smiles'
            })
            
            dfs.append(sample_data_frame)
    
    # Concatenate all data into one DataFrame
    big_frame = pd.concat(dfs, ignore_index=True, axis=0)
    big_frame.to_csv(output_csv, index=False)
    print(f"Successfully created {output_csv} with {len(big_frame)} rows")


if __name__ == '__main__':
    main()