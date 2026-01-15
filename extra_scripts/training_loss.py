#!/usr/bin/env python 
# Generates a .png plot to visualize model training and validation loss by epoch
# NOTE: THIS SCRIPT NEEDS REVISION. SEE BELOW

###############
### IMPORTS ###
###############
import pandas as pd
import matplotlib.pyplot as plt

############
### BODY ###
############

log_file = "../setup_model/tlearn_HILIC_41262880.out.log" ## CHANGE THIS TO AN ARGPARSE SO I CAN RUN THIS EVERYTIME WE TRAIN ! 

train_stats = {}
valid_stats = {}

with open(log_file, "r") as f:
    for line in f:
        if "train | epoch" in line: # FIX THIS INEFFICIENT LOOP
            segments = line.split("|")
            epoch_segment = [s for s in segments if "epoch" in s][0]
            epoch_num = int(epoch_segment.split("epoch")[1].strip())
            
            for seg in segments:
                seg = seg.strip()
                if seg.startswith("loss "):
                    train_stats[epoch_num] = float(seg.split("loss")[1].strip())

        if "valid | epoch" in line:
            segments = line.split("|")
            epoch_segment = [s for s in segments if "epoch" in s][0]
            epoch_num = int(epoch_segment.split("epoch")[1].strip())
            
            for seg in segments:
                seg = seg.strip()
                if seg.startswith("loss "):
                    valid_stats[epoch_num] = float(seg.split("loss")[1].strip())

epochs = sorted(train_stats.keys())

# THIS IS INEFFICIENT
df = pd.DataFrame({
    "Epoch": epochs,
    "Train_Loss": [train_stats[e] for e in epochs],
    "Valid_Loss": [valid_stats[e] for e in epochs]})

##############
### FIGURE ###
##############
plt.plot(df['Epoch'], df['Train_Loss'], color='tomato', label='Training Loss')
plt.plot(df['Epoch'], df['Valid_Loss'], color='dodgerblue', label='Validation Loss')
plt.xlabel("Epoch", fontsize="x-large")
plt.ylabel("Model Loss", fontsize="x-large")
plt.title("TRAINED MODEL LOSS BY EPOCH (e=250)", fontsize="xx-large", fontweight="bold")
plt.legend(fontsize="medium")
fig = plt.gcf()
plt.savefig('Training_Loss.png', dpi=fig.dpi) # CH
plt.close(fig)