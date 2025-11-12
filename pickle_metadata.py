#!/usr/bin/env python 

import pandas as pd
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Process a file and save the output in a TSV."
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

df = pd.read_csv(args.input_file, sep='\t')

df.to_pickle(args.output_file)

print(f"Successfully converted '{args.input_file}' to '{args.output_file}'")