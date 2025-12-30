#!/usr/bin/env python 
from pathlib import Path
import pandas as pd

#NOTE: THIS SCRIPT EXCLUDES DATA WITHOUT SMILES (this reduced HILIC_negative_data from 86MB to 6.3MB)

#folder = Path('HILIC_negative_data')
#output_csv = 'HILIC_negative_data.csv'
folder = Path('HILIC_positive_data')
output_csv = 'HILIC_positive_data.csv'

file_count = sum(1 for item in folder.iterdir() if item.is_file())

dfs = []
#for data_id in range(1,file_count+1):
for data_id in range(303,303+file_count+1):
    padded_id = f"{data_id:04d}"
    matched_files = list(folder.glob(f'*_{padded_id}')) #search for file ending with _{data_id}
    for file in matched_files:
        print(f"Extracting {file.name}...")

        sample_data_frame = pd.read_csv(folder/file.name)
        sample_data_frame['File ID'] = padded_id #add row of file id
        sample_data_frame = sample_data_frame[['File ID', 'Mass Feature ID', 'Retention Time (min)', 'smiles']].dropna(how='any', axis=0) #trim rows without smiles and extract only select columns
        dfs.append(sample_data_frame) 
        #print(dfs)

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True, axis=0)

big_frame.to_csv(output_csv, index=False)