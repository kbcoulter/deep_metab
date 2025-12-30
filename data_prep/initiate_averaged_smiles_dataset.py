#!/usr/bin/env python 
import csv

input_file_path = "emp_500_RP_data.csv"
output_file_path = 'emp_500__averaged_smiles_for_graphormer.csv'

averaged_smiles = {}
id_str = "0001"

with open(input_file_path, 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    
    next(reader) #skip header line

    for row in reader:
        file_id = row[0]              # Col 1 --> fileID
        total_rt = float(row[2])      # Col 3 --> retention time (min)
        smiles = row[3]               # Col 4 --> smiles

        if smiles in averaged_smiles:
            averaged_smiles[smiles][0] += total_rt
            averaged_smiles[smiles][1] += 1
        else:
            averaged_smiles[smiles] = [total_rt, 1]

# Write to Output File
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for smiles, values in averaged_smiles.items():
        avg_rt = values[0] / values[1] #calculate average retention time
        
        outfile.write(f"{id_str},{smiles},{avg_rt}\n")