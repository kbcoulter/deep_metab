#!/usr/bin/env python 

import pickle
import pandas as pd
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Print the contents of a pickle file.',
        help='Simply prints the contents of a pickle file to stdout.'
    )
    
    parser.add_argument('input_file', type=str,
                        help='Path to the pickle file to unpack and print')
    
    args = parser.parse_args()
    file_name = args.input_file
    
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


if __name__ == '__main__':
    main()
