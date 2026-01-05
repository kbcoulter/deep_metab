#!/usr/bin/env python

# necessary imports
import pandas as pd
import os
import numpy as np
import argparse
from tqdm import tqdm

# argparse argument
# TODO: Fill out description/ Help Message
def get_args():
    parser = argparse.ArgumentParser(description= "Append or Analyse Existing Files")
    parser.add_argument("-i", "--input", help="Input Directory with LC-MS Data (Raw or Appended)",
                        required= True)
    parser.add_argument("-p", "--preds", help="Input the smiles and predictions csv file.",
                        default=None)
    parser.add_argument("-a", "--append", action="store_true",
                        help="Option to utilize the dictionary to look\
                              up retention times from smiles and append \
                              it to the LC-MS data in a new file")
    parser.add_argument("-t", "--type", help="Option to specify either `hilic or rp` for the sake\
                        of naming the output files later in the script",
                        choices=['hilic', 'rp'], required=True)
    parser.add_argument('-s', '--stats', action="store_true", help="Option to perform analysis and print out stats")
    parser.add_argument("--save", action="store_true", help="Choose to save output or not.")

    return parser.parse_args()

# Set up global variables
args = get_args()

dir_path = args.input
smiles_csv = args.preds
data_type = args.type

# Boolean Vars
append_switch = args.append
stats_switch = args.stats
save_switch = args.save

# before analysis
ambiguous_count_0 = 0
unambiguous_count_0 = 0
# after analysis
ambiguous_count_1 = 0
unambiguous_count_1 = 0

# set up the dictionary if containing a predictions csv file
if smiles_csv != None:
    try:
        smiles_df = pd.read_csv(smiles_csv)
        smiles_dict = smiles_df.set_index('SMILES')['Predicted RT'].to_dict()
    except FileNotFoundError:
        print(f"Error: File '{smiles_csv}' not found.") 

# Parse the file
try:
    for file in tqdm(os.listdir(dir_path), desc="Processing CSV Files"):
        file_path = os.path.join(dir_path, file) # get file path

        if os.path.isfile(file_path) == False: 
            # make sure it's a file and not a dir
            print("Exiting. Script will Not Process a Directory. Please Include Only Files!")
            break
        # load the dataframe
        #print(f'Opening File: {file_path}')
        df_csv = pd.read_csv(file_path)
        # only work with rows with smiles for now
        # keeping some NAs will be a little tricky because it will count one occurence of mass feature id as unambiguous even if there's no smiles (we dont want that)
        df_csv = df_csv.dropna(subset=[df_csv.columns[31]]) # drop columns without smile

        # map values from dict if append is true
        if append_switch:
            df_csv['predicted_rt'] = df_csv['smiles'].map(smiles_dict) # col index 38
        
        # store number of (un)ambigous counts before
        if stats_switch:
            # Get Before Counts
            counts = df_csv['Mass Feature ID'].value_counts() 
            ambiguous_count_0 += (counts > 1).sum() # mf_id appearing multiple times
            unambiguous_count_0 += (counts == 1).sum() # mf_id appearing only once
    
            # get absolute diff for rt
            df_csv['abs_diff_rt'] = abs(df_csv['predicted_rt'] - (df_csv['Retention Time (min)']*60)) # col idx 39

            # perform ranking - use dense method to give dupes the same ranking
            df_csv["rank_mz_err"] = df_csv.groupby('Mass Feature ID')['m/z Error Score'].rank(method="dense", ascending=True)

            df_csv["rank_entropy"] = df_csv.groupby('Mass Feature ID')['Entropy Similarity'].rank(method="dense", ascending=False)

            df_csv["rank_delta_rt"] = df_csv.groupby('Mass Feature ID')['abs_diff_rt'].rank(method="dense", ascending=True)

            # Get After Counts 
            # cond1: mass feature id only appeared once (automatically unambiguos)
            id_counts = df_csv['Mass Feature ID'].value_counts()
            single_occurrence_ids = id_counts[id_counts == 1].index
            # cond 2: if it has multiple columns so find row with (1,1,1) ranking
            cond = (df_csv["rank_mz_err"] == 1) & (df_csv["rank_entropy"] == 1) & (df_csv["rank_delta_rt"] == 1)
            has_111_per_id = cond.groupby(df_csv['Mass Feature ID']).any()
            ids_with_111 = has_111_per_id[has_111_per_id].index
            # retrieve ids
            unambiguous_ids = set(single_occurrence_ids) | set(ids_with_111)
            ambiguous_ids = set(id_counts[id_counts > 1].index) - set(ids_with_111)
            # print(len(ambiguous_ids))
            # print(ids_with_111)
            unambiguous_count_1 += len(unambiguous_ids)
            ambiguous_count_1 += len(ambiguous_ids)
        
        #print(f'Closing File: {file_path}')
        # save appended df to a new csv
        #break # break early
        if save_switch:
            df_csv.to_csv(f'{file}_appended.csv')

    # Write the Report
    if stats_switch:
        total_counts = ambiguous_count_0 + unambiguous_count_0
        pct_ambiguous_0 = np.round(ambiguous_count_0*100/total_counts, decimals=2)
        pct_unambiguous_0 = np.round(unambiguous_count_0*100/total_counts, decimals=2)

        diff_ambiguous = ambiguous_count_1 - ambiguous_count_0
        diff_unambiguous = unambiguous_count_1 - unambiguous_count_0
        # commpute percetnage change
        pct_ambiguous_1 = np.round(ambiguous_count_1*100/total_counts, decimals=2)
        pct_unambiguous_1 = np.round(unambiguous_count_1*100/total_counts, decimals=2)

        # Aditional Feature Track Which Ids Are Ambiguous and Print Out their details
        # output a csv file with ambiguous the ids

        with open(f'{data_type}_stats_report.txt', 'w') as fout:
            fout.write(f"Before Applying Graphormer_RT:\n\
                    Number of Ambiguous Annotations: {ambiguous_count_0} ({pct_ambiguous_0}%)\n\
                    Number of Unambiguous Annotations: {unambiguous_count_0} ({pct_unambiguous_0}%) \n\n\
After Applying Graphormer_RT:\n\
                    Number of Ambiguous Annotations: {ambiguous_count_1} ({pct_ambiguous_1}%)\n\
                    Number of Unambiguous Annotations: {unambiguous_count_1} ({pct_unambiguous_1}%)\n\n\
Stats Changes:\n\
                    Change in Ambiguous Counts: {diff_ambiguous} | Relative Ambiguous Change: {np.round(diff_ambiguous*100/ambiguous_count_0, decimals=2)}%\n\
                    Change in Unmbiguous Counts: {diff_unambiguous} | Relative Unambiguous Change: {np.round(diff_unambiguous*100/unambiguous_count_0, decimals=2)}%")
        
# Errors out to filenotfound error
except FileNotFoundError:
    print(f"Error: Directory '{dir_path}' not found.") 

