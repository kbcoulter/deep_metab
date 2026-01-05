#!/usr/bin/env python 
import csv
import os
import urllib.request

input_file="/projects/bgmp/shared/groups/2025/deepmetab/ewi/deep_metab/bioscales_HILIC_pos_urls.csv"
#input_file="/projects/bgmp/shared/groups/2025/deepmetab/ewi/deep_metab/bioscales_HILIC_neg_urls.csv"
output_dir="HILIC_positive_data"
#output_dir="HILIC_negative_data"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(input_file, 'r') as f:
    reader = csv.reader(f)
    next(reader, None) #skip header

    #counter = 1
    counter = 303
    for row in reader:
            if len(row) != 2:
                print("incomplete row, skipping...")
                continue
                           
            data_id = row[0]
            url = row[1]
            filename = f"{data_id}_{counter:04d}"    

            file_path = os.path.join(output_dir, filename)
            
            print(f"Downloading {data_id} as {filename}...")
            
            urllib.request.urlretrieve(url, file_path)
            
            counter += 1



