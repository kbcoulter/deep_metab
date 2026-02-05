#!/usr/bin/env python 
import argparse

parser = argparse.ArgumentParser(
    description="Output the difference between True RT and Predicted RT."
)

parser.add_argument(
    "--input_file",
    '-i',
    type=str,  
    required=True,
    help="The input file to process.",
)

parser.add_argument(
    "--output_file",
    '-o',
    type=str,  
    default="difference.txt",
    help="The input file to process.",
)

args = parser.parse_args()


#print
with open(args.input_file, 'r') as preds, open(args.output_file,'w') as preds_difference:
    preds_difference.write(f"Raw Diff\t% Diff:")
    next(preds) #skip the first line

    num_total=0
    num_less_than_10=0
    num_less_than_20=0
    num_less_than_30=0
    num_less_than_40=0
    num_less_than_50=0
    smiles_less_than_10 = []
    smiles_10_to_20 = []
    smiles_20_to_30 = []
    smiles_30_to_40 = []
    smiles_40_to_50 = []

    for line in preds:
        parts = line.split(',')
        smiles = parts[0].strip()

        true_rt = float(parts[2])
        predicted_rt = float(parts[3])
        difference = abs(true_rt - predicted_rt)
        percentage_difference = (abs(true_rt - predicted_rt) / true_rt) * 100
        if(percentage_difference<=10):
            num_less_than_10+=1
            smiles_less_than_10.append(smiles)
        elif(percentage_difference<=20):
            num_less_than_20+=1
            smiles_10_to_20.append(smiles)
        elif(percentage_difference<=30):
            num_less_than_30+=1
            smiles_20_to_30.append(smiles)
        elif(percentage_difference<=40):
            num_less_than_40+=1
            smiles_30_to_40.append(smiles)
        elif(percentage_difference<=50):
            num_less_than_50+=1

        preds_difference.write(f"{smiles}\t{difference:.4f}\t{percentage_difference:.2f}%\n")
        num_total+=1

    preds_difference.write(f"\n\nTOTAL PREDICTIONS: {num_total}\n\n")

    preds_difference.write(f"TOTAL PREDICTIONS WITH A % DIFF <= 10: {num_less_than_10}\n")
    preds_difference.write(f"TOTAL PREDICTIONS WITH A % DIFF BETWEEN 10,20: {num_less_than_20}\n")
    preds_difference.write(f"TOTAL PREDICTIONS WITH A % DIFF BETWEEN 20,30: {num_less_than_30}\n")
    preds_difference.write(f"TOTAL PREDICTIONS WITH A % DIFF BETWEEN 30,40: {num_less_than_40}\n")
    preds_difference.write(f"TOTAL PREDICTIONS WITH A % DIFF BETWEEN 40,50: {num_less_than_50}\n\n")

    preds_difference.write(f"PROPORTION OF PREDICTIONS WITH A % DIFF <= 10: {num_less_than_10/num_total*100}%\n")
    preds_difference.write(f"PROPORTION OF PREDICTIONS WITH A % DIFF BETWEEN 10,20: {num_less_than_20/num_total*100}%\n")
    preds_difference.write(f"PROPORTION OF PREDICTIONS WITH A % DIFF BETWEEN 20,30: {num_less_than_30/num_total*100}%\n")
    preds_difference.write(f"PROPORTION OF PREDICTIONS WITH A % DIFF BETWEEN 30,40: {num_less_than_40/num_total*100}%\n")
    preds_difference.write(f"PROPORTION OF PREDICTIONS WITH A % DIFF BETWEEN 40,50: {num_less_than_50/num_total*100}%\n\n")    

    preds_difference.write("SMILES STRINGS BY PERCENTAGE DIFFERENCE INTERVAL\n")
    preds_difference.write("="*50 + "\n")

    preds_difference.write("\nSMILES with % Diff <= 10:\n")
    preds_difference.write("\n".join(smiles_less_than_10) + "\n")

    preds_difference.write("\nSMILES with % Diff BETWEEN 10 and 20:\n")
    preds_difference.write("\n".join(smiles_10_to_20) + "\n")

    preds_difference.write("\nSMILES with % Diff BETWEEN 20 and 30:\n")
    preds_difference.write("\n".join(smiles_20_to_30) + "\n")

    preds_difference.write("\nSMILES with % Diff BETWEEN 30 and 40:\n")
    preds_difference.write("\n".join(smiles_30_to_40) + "\n")

    preds_difference.write("\nSMILES with % Diff BETWEEN 40 and 50:\n")
    preds_difference.write("\n".join(smiles_40_to_50) + "\n")