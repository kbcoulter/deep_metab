#!/usr/bin/env python 

import argparse
import csv
from pathlib import Path

####################
#INPUTS AND OUTPUTS#
####################


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
    "--data_id",
    '-d',
    type=str, 
    required=True,
    help="The data id of the sample run.",
)


#Get input file
temp_args, _ = parser.parse_known_args()
#Initialize a default output filepath
default_output_file = temp_args.input_file.with_suffix(".tsv")

# Add the output file argument with the calculated default
parser.add_argument(
    "--output_file",
    '-o',
    type=Path,
    default=default_output_file,
    help="The output cleaned file. Defaults to the input filename with a .tsv extension.",
)

args = parser.parse_args()

#############
#SCRIPT BODY#
#############

FINAL_COLUMN_ORDER = [
    'id',
    'name',
    'formula',
    'rt',
    'pubchem.cid',
    'pubchem.smiles.isomeric',
    'pubchem.smiles.canonical',
    'pubchem.inchi',
    'pubchem.inchikey',
    'id.chebi',
    'id.hmdb',
    'id.lipidmaps',
    'id.kegg',   
]


# A row is kept if it has ALL REQUIRED_FIELDS *and* AT LEAST ONE CONDITIONAL_FIELD.
with open(args.input_file, 'r') as csv_file, open(args.output_file,'w') as cleaned_tsv_file:
    mapping_rules = [
        ('pubchem.smiles.canonical', 'pubchem.smiles.canonical'),
        ('pubchem.cid', 'pubchem.cid'),
        ('smiles', 'pubchem.smiles.isomeric'),
        ('inchikey', 'pubchem.inchikey'),
        ('inchi', 'pubchem.inchi'),
        ('lipidmaps', 'id.lipidmaps'),
        ('chebi', 'id.chebi'),
        ('hmdb', 'id.hmdb'),
        ('kegg', 'id.kegg'),
        ('cid', 'pubchem.cid'), 
    ]

    reader = csv.DictReader(csv_file)
    
    original_headers = reader.fieldnames
    if not original_headers:
        print("Error: CSV file has no headers.")
        
    new_header_map = {} # Stores { 'Original Header': 'new_name' }
    assigned_new_names = set() # Stores { 'new_name' } to check for duplicates
    
    #CREATING MAP OF ORIGINAL HEADERS TO NEW ONES 
    for header in original_headers:
        if not header: # Handle potential empty header names
            continue
            
        normalized_header = header.lower().replace(" ", "")
        new_name = None
        
        # 1. Check special conditions first
        if normalized_header.startswith('rt') or 'r/t' in normalized_header or 'retentiontime' in normalized_header:
            new_name = 'rt'
        elif normalized_header == 'id' or (normalized_header.startswith('mass') and 'id' in normalized_header):
            new_name = 'id' 
        elif normalized_header.startswith('mol') and 'formula' in normalized_header:
            new_name = 'formula'
        elif 'name' in normalized_header and (normalized_header.startswith('mol') or 'name' == normalized_header):
            new_name = 'name'
        else:
            # 2. Check prioritized list
            for substring, mapping in mapping_rules:
                if substring in normalized_header:
                    new_name = mapping
                    break # Found the first, highest-priority match
        
        # If a rule matched...
        if new_name:
            # Check for conflicts
            if new_name in assigned_new_names:
                # Find which *other* original header already mapped to this
                conflicting_header = None # Initialize a variable to hold the name
                # Iterate through all the original_header -> new_name pairs we've already mapped
                for original_header, mapped_name in new_header_map.items():
                    # Check if the mapped_name is the one that's causing our current conflict
                    if mapped_name == new_name:
                        # If it is, this 'original_header' is the one we're looking for
                        conflicting_header = original_header
                        # We found it, so we can stop searching
                        break
                # Raise an error and break
                raise ValueError(f"Conflict: Both '{header}' and '{conflicting_header}' map to '{new_name}'.")
            
            new_header_map[header] = new_name
            assigned_new_names.add(new_name)
        
    # --- 3. Validation Step ---
    print("--- Header Validation Report ---")
    print(f"Original headers found: {original_headers}")
    print(f"Remapped headers: {new_header_map}")
    print(f"Final set of new keys: {assigned_new_names}\n")

    is_valid = True

    # Requirement 1: Core fields
    core_fields = {'id', 'name', 'formula', 'rt'}
    missing_core = core_fields - assigned_new_names
    if missing_core:
        print(f"FAILED: Missing required core fields: {missing_core}")
        is_valid = False
    else:
        print("SUCCESS: All core fields found (id, name, formula, rt).")
        
    # Requirement 2: At least one ID field
    id_fields = {
        'pubchem.cid',
        'pubchem.smiles.isomeric',
        'id.chebi',
        'id.hmdb',
        'id.lipidmaps',
        'id.kegg'
    }
    
    if not assigned_new_names.intersection(id_fields):
        print(f"FAILED: Missing at least one key identifier from the set: {id_fields}")
        is_valid = False
    else:
        print("SUCCESS: At least one key identifier (e.g., pubchem.cid, smiles) found.")
    
    if not is_valid:
        raise ValueError("Sorry, no numbers below zero are allowed.")
    


    # BUILDING THE NEW TSV FILE 

    # Invert the header mappings dict for easier retrieval
    inverted_remapped = {value: key for key, value in new_header_map.items()}
    
    # Writing column headers 
    cleaned_tsv_file.write('\t'.join(FINAL_COLUMN_ORDER) + '\n')

    # For each row 
    for original_row in reader:
        # Initialize row values that will get printed
        FINAL_ROW_VALUES = []
        # Map original values to the final column order of the tsv
        for column_name in FINAL_COLUMN_ORDER:
            original_key = inverted_remapped.get(column_name)
            if original_key and original_key in original_row:
                value = original_row[original_key]
                
                # Prepend the data ID (experiment) to the sample ID
                if column_name == 'id' and value:
                    value = f"{args.data_id}_{value}"

                FINAL_ROW_VALUES.append(value)
            else:
                FINAL_ROW_VALUES.append('')

        # Conditional for sample data: check Monotopic Mass Feature ID does not = Mass Feature ID
        mono_id = original_row.get("Monoisotopic Mass Feature ID")
        mass_id = original_row.get("Mass Feature ID")
        if (mono_id is not None and mono_id != '') and mono_id != mass_id:
            #print(f"Skipping row (different IDs): {original_row}")
            continue

        # Check all necessary fields and at least one conditional field is satisfied
        if all(FINAL_ROW_VALUES[i] for i in range(4)) and any(FINAL_ROW_VALUES[i] for i in range(4, 10)):
            # Print the values to new tsv file
            cleaned_tsv_file.write('\t'.join(FINAL_ROW_VALUES) + '\n')


###################
# sample mappings #
###################

# FINAL_COLUMN_ORDER = [
#     'apple',
#     'orange',
#     'banana',
#     'kiwi',
#     'watermelon'
# ]
# remapped = {
#     'human':'apple',
#     'hammer':'kiwi',
#     'lamp':'banana',
#     'couch':'orange'
# }
# original = {
#     'human': 1,
#     'lamp': 'tree',
#     'hammer': '',
#     'couch':0      
# }
# FINAL_RESULT = [1,0,'tree','',''] 

    

            
