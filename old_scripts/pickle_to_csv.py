#!/usr/bin/env python

import pickle
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Convert a metadata pickle file (from python dictionary) to a csv.")

default_input = '/projects/bgmp/shared/groups/2025/deepmetab/ewi/deep_metab/sample_data_0001/RP_metadata.pickle' 

parser.add_argument(
    "-i", "--input",
    dest="pickle_path",
    default=default_input,
    help=f"Input pickle file path. Default: {default_input}")

parser.add_argument(
    "-o", "--output",
    dest="csv_path",
    default="cleaned_data.csv",
    help="Output csv file path. Default: cleaned_data.csv")

args = parser.parse_args()

with open(args.pickle_path, 'rb') as f:
    data = pickle.load(f)

df = pd.DataFrame(data)
df.to_csv(args.csv_path, index=False)

print(f"Un-Pickled to: {args.csv_path}")
