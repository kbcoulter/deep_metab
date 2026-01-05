#!/usr/bin/env python 

import pickle
import pandas as pd
import sys

# Define the name of your pickle file
file_name = '/projects/bgmp/shared/groups/2025/deepmetab/ewi/deep_metab/sample_data_from_graphormer/RP_metadata.pickle'

try:
    # Open the file in 'read binary' (rb) mode
    with open(file_name, 'rb') as f:
        # Load the data from the file
        data = pickle.load(f)
        
        # Print the data
        print("Successfully loaded data:")
        print(data)

except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found.")
except pickle.UnpicklingError:
    print(f"Error: Failed to unpickle. The file '{file_name}' may be corrupted or not a pickle file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
