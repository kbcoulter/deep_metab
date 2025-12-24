#!/usr/bin/env python 
import pandas as pd

# 1. Load the datasets
# 'df_main' is the first dataset you want to change (File 1)
# 'df_dict' is the second dataset acting as the dictionary (File 2)
df_main = pd.read_csv('emp_500_RP_data.csv') 
df_dict = pd.read_csv('/projects/bgmp/shared/groups/2025/deepmetab/ewi/deep_metab/predictions_RP/40581193.csv')

# 2. Rename the observed retention time column in the main dataset
df_main = df_main.rename(columns={'Retention Time (min)': 'Observed Retention Time (min)'})

# 3. Prepare the dictionary data
# We only need the SMILES and Predicted RT columns from the second file.
# We also drop duplicates on 'SMILES' to ensure it acts like a true 1:1 dictionary.
lookup_table = df_dict[['SMILES', 'Predicted RT']].drop_duplicates(subset=['SMILES'])

# 4. Merge the datasets
# We use a "left" merge. This keeps every row in your main dataset (df_main) 
# and attaches the matching info from the lookup_table.
# Note: We match 'smiles' from the left df to 'SMILES' from the right df.
merged_df = pd.merge(df_main, lookup_table, left_on='smiles', right_on='SMILES', how='left')

# 5. Create the new column and convert seconds to minutes
merged_df['Predicted Retention Time (min)'] = merged_df['Predicted RT'] / 60

# 6. Reorder the columns
# You wanted the new column between column 3 and 4.
# Based on your screenshot, that is between "Observed Retention Time (min)" and "smiles".
cols = [
    'File ID', 
    'Mass Feature ID', 
    'Observed Retention Time (min)', 
    'Predicted Retention Time (min)', 
    'smiles'
]

# Create the final dataframe with only the columns we want, in the order we want
final_df = merged_df[cols]

# 7. Save to a NEW file (so we don't mutate the original)
final_df.to_csv('final_emp_500_data.csv', index=False)

print("Process complete! Check 'final_emp_500_data.csv' for the results.")