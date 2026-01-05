#!/usr/bin/env python 

#import pandas as pd
import argparse
from pathlib import Path
import pickle
import csv

parser = argparse.ArgumentParser(
    description="Process a file and save the output in a pickle. Expects a csv."
)

parser.add_argument(
    "--input_file",
    '-i',
    type=Path,  
    required=True,
    help="The input file to process.",
)

parser.add_argument(
    "--output_file",
    '-o',
    type=Path,  
    required=True,
    help="The output file to process.",
)

args = parser.parse_args()

# df = pd.read_csv(args.input_file, sep='\t')

# df.to_pickle(args.output_file)

# print(f"Successfully converted '{args.input_file}' to '{args.output_file}'")
metadata_dict = {}
print(f"Opening input file: {args.input_file}")
try:
    with open(args.input_file, mode='r', encoding='utf-8') as f:
        # Use csv.reader to handle TSV
        reader = csv.reader(f, delimiter='\t')
        try:
            # Read and print the header
            header = next(reader)
            print(f"Read header: {header}")
        except StopIteration:
            print("File is empty.")
            header = []
        
        # Process the rest of the rows
        count = 0
        for row in reader:
            if row: # ensure row is not empty
                key = row[0]       # First column is the key
                value = row[1:]    # The rest of the columns are the value list
                metadata_dict[key] = value
                count += 1
        
        print(f"Processed {count} rows into dictionary.")

    if not metadata_dict:
        print(f"Warning: No data was read from {args.input_file}. Output pickle will be an empty dictionary.")

    print(f"Saving dictionary to pickle file: {args.output_file}")
    with open(args.output_file, 'wb') as f_out:
        # Use pickle.dump to save the dictionary
        pickle.dump(metadata_dict, f_out)
    
    print(f"Successfully converted '{args.input_file}' to '{args.output_file}'")

except FileNotFoundError:
    print(f"Error: Input file not found at {args.input_file}")
except Exception as e:
    print(f"An error occurred: {e}")