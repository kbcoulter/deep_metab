#!/usr/bin/env python

import argparse
from sklearn.model_selection import train_test_split
import pandas as pd

parser = argparse.ArgumentParser(description='Split HILIC "Known" Data (.csv) into Testing and Training Sets (.csv) Using a Proportion and Random Seed')

parser.add_argument('-d',"--data", type=str, required=True, help="Input data.csv file to be split, where each row is an entry.")
# Cut default path from /projects/bgmp/shared/groups/2025/deepmetab/kcoulter/deep_metab/my_data/HILIC_knowns to save confusion
parser.add_argument('-r', "--rseed", type=int, required=False, default=26, help="Random seed to maintain reproducibility. Defaults to 26.")
parser.add_argument('-s', '--split', type = float, required = False, default = 0.1, help = "Proportion to split training and test data. Defautls to 0.1.")
parser.add_argument('-o', '--out', type = str, required = False, default = None, help = "A prefix for train and test data out: <out>_test.csv")

args = parser.parse_args()

train = "train.csv"
test = "test.csv"
out = args.out
if out:
    train = args.out + "_" + train
    test = args.out + "_" + test

df = pd.read_csv(args.data, dtype={0: str})

train_df, test_df = train_test_split(
    df,
    test_size=args.split,
    random_state=args.rseed,
    shuffle=True)

train_df.to_csv(train, index=False)
test_df.to_csv(test, index=False)

print(f"From {args.data}:")
print(f"Saved {len(train_df)} rows as {train}")
print(f"Saved {len(test_df)} rows as {test}")
