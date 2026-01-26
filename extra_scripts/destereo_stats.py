#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

# Read original and destereoed files
original_smiles = []
destereoed_smiles = []

with open('output.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if len(row) >= 4:
            original_smiles.append(row[3])

with open('data_without_stereo.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if len(row) >= 4:
            destereoed_smiles.append(row[3])

# Count how many original SMILES map to each destereoed SMILES
collapse_map = defaultdict(list)
for orig, dest in zip(original_smiles, destereoed_smiles):
    collapse_map[dest].append(orig)

# Helper to check if SMILES has stereochemistry markers
def has_stereochemistry(smiles):
    """Check if SMILES contains stereochemistry markers (@, /, \)"""
    return '@' in smiles or '/' in smiles or '\\' in smiles

# Statistics
unique_original = len(set(original_smiles))
unique_destereoed = len(set(destereoed_smiles))
collapsed_count = sum(1 for v in collapse_map.values() if len(set(v)) > 1)

# Calculate total original SMILES accounted for by collapses
total_collapsed_originals = 0
for dest, origs in collapse_map.items():
    unique_origs = len(set(origs))
    if unique_origs > 1:
        total_collapsed_originals += unique_origs

# Calculate reduction from collapses vs canonicalization
reduction_from_collapses = total_collapsed_originals - collapsed_count
reduction_from_canonicalization = (unique_original - unique_destereoed) - reduction_from_collapses

print(f"Original unique SMILES: {unique_original:,}")
print(f"Destereoed unique SMILES: {unique_destereoed:,}")
print(f"Reduction: {unique_original - unique_destereoed:,} ({100*(1-unique_destereoed/unique_original):.1f}% reduction)")
print(f"\nDestereoed SMILES with multiple original forms: {collapsed_count:,}")
reduction_total = unique_original - unique_destereoed
print(f"\nBreakdown of reduction:")
if reduction_total > 0:
    print(f"  From stereochemistry collapses: {reduction_from_collapses:,} ({100*reduction_from_collapses/reduction_total:.1f}%)")
    print(f"  From canonicalization (non-stereo): {reduction_from_canonicalization:,} ({100*reduction_from_canonicalization/reduction_total:.1f}%)")
else:
    print(f"  From stereochemistry collapses: {reduction_from_collapses:,}")
    print(f"  From canonicalization (non-stereo): {reduction_from_canonicalization:,}")

# Analyze collapses: which ones involve stereochemistry?
stereo_collapses = 0
canon_collapses = 0
for dest, origs in collapse_map.items():
    unique_origs = list(set(origs))
    if len(unique_origs) > 1:
        # Check if any of the originals had stereochemistry markers
        has_stereo = any(has_stereochemistry(orig) for orig in unique_origs)
        if has_stereo:
            stereo_collapses += 1
        else:
            canon_collapses += 1

print(f"\nCollapse analysis:")
print(f"  Collapses involving stereochemistry markers: {stereo_collapses:,}")
print(f"  Collapses from canonicalization only: {canon_collapses:,}")

# Show distribution of collapse sizes
collapse_sizes = defaultdict(int)
for dest, origs in collapse_map.items():
    unique_origs = len(set(origs))
    if unique_origs > 1:
        collapse_sizes[unique_origs] += 1

print("\nCollapse size distribution:")
for size in sorted(collapse_sizes.keys()):
    print(f"  {size} original SMILES → 1 destereoed: {collapse_sizes[size]:,} cases")

# Show top examples
print("\nTop 10 examples of collapse:")
sorted_collapses = sorted(collapse_map.items(), key=lambda x: len(set(x[1])), reverse=True)
for dest, origs in sorted_collapses[:10]:
    unique_origs = list(set(origs))
    if len(unique_origs) > 1:
        print(f"\nDestereoed: {dest}")
        print(f"  Collapsed from {len(unique_origs)} unique SMILES:")
        for orig in unique_origs[:5]:  # show first 5
            print(f"    - {orig}")
        if len(unique_origs) > 5:
            print(f"    ... and {len(unique_origs)-5} more")