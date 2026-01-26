#!/usr/bin/env python 

import argparse
import csv
from pathlib import Path

from rdkit import Chem


def has_stereochemistry(smiles: str) -> bool:
    """Check if SMILES contains stereochemistry markers (@, /, \\)."""
    return '@' in smiles or '/' in smiles or '\\' in smiles


def remove_stereochemistry(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def process_csv(
    input_csv: Path, output_csv: Path, smiles_col_index: int, has_header: bool
) -> None:
    """Read a CSV, remove stereochemistry in the specified column, and write a new CSV."""
    with input_csv.open(newline="") as infile, output_csv.open("w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row_number, row in enumerate(reader, start=1):
            if has_header and row_number == 1:
                # Add the new column header
                #row.append('Originally a Stereoisomer') TEST CASE, CHECK IF remove_stereochemistry only changes stereoisomers
                #row.append('Smiles Changed')
                row.append('smiles_destereo')
                row.append('is_stereo')
                writer.writerow(row)
                continue

            if smiles_col_index >= len(row):
                raise IndexError(
                    f"Row {row_number} has no column at index {smiles_col_index}"
                )

            smiles = row[smiles_col_index]

            # Store original SMILES for comparison
            #original_smiles = smiles
            
            # Check if original SMILES had stereochemistry before processing
            was_stereoisomer = has_stereochemistry(smiles)
            
            try:
                processed_smiles = remove_stereochemistry(smiles)
                # Keep original SMILES and add destereoed version as new column
                #row[smiles_col_index] = processed_smiles
                # Check if SMILES changed after processing
                #smiles_changed = (original_smiles != processed_smiles) TEST CASE, CHECK IF remove_stereochemistry only changes stereoisomers
            except ValueError as exc:
                raise ValueError(f"Error on row {row_number}: {exc}") from exc

            # Append True/False for whether SMILES changed
            #row.append(smiles_changed) TEST CASE, CHECK IF remove_stereochemistry only changes stereoisomers
            
            # Append the destereoed SMILES
            row.append(processed_smiles)

            # Append True/False for whether it was originally a stereoisomer
            row.append(was_stereoisomer)

            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove stereochemistry from a SMILES column in a CSV file."
    )
    parser.add_argument(
        "--input-csv",
        '-i',
        required=True,
        type=Path,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output-csv",
        '-o',
        required=True,
        type=Path,
        help="Path to write the output CSV file.",
    )
    parser.add_argument(
        "--smiles-col-index",
        '-col',
        required=True,
        type=int,
        help="Zero-based column index containing SMILES strings.",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Treat the first row as a header and copy it through unchanged.",
    )

    args = parser.parse_args()
    process_csv(args.input_csv, args.output_csv, args.smiles_col_index, args.has_header)


if __name__ == "__main__":
    main()
