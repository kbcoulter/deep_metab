#!/usr/bin/env python 
import csv
import argparse

# set up argparse
def get_args():
    parser = argparse.ArgumentParser(description="Reformat Metadata File Correctly.")
    parser.add_argument('-t', '--template', help="Specify the template file.", required=True)
    parser.add_argument('-i', '--input', help="Specify the file to be reformatted.", required=True)
    parser.add_argument('-o', '--output', help="Specify an output file name.")
    parser.add_argument('-d', '--dir', help="Specify Output Directory")
    # parser.add_argument('-id', help="Specify ID of metadata file")

    return parser.parse_args()

args = get_args()
# specify path to files
template = args.template
input = args.input
output = args.output
dir = args.dir

with open(template, newline="") as f1, open(input, newline="") as f2:
    r1 = list(csv.reader(f1, delimiter="\t"))
    r2 = list(csv.reader(f2, delimiter="\t"))

header = r1[0]
row1_template = r1[1]   # first data row in template (used as template)

# quick lil sanity check: header alignment
assert header == r2[0], "Headers in Template and Input do not match!"

# look for how '' is represented in og template
missing_as_blank_cols = [
    i for i, v in enumerate(row1_template) if v == ""
]

print("Columns that use empty string for missing values:")
for i in missing_as_blank_cols:
    print(i, header[i])

fixed_rows = []
fixed_rows.append(header)  # keep header

# and so begins the fix
for row in r2[1:]:
    row = row[:] 
    for i in missing_as_blank_cols:
        if row[i] == "0":
            row[i] = ""
    fixed_rows.append(row)

# write out the new second file
with open(f'{dir}/{output}/{output}_clean_metadata.tsv', "w", newline="") as out:
    w = csv.writer(out, delimiter="\t")
    w.writerows(fixed_rows)

print("Wrote corrected file to", output)
