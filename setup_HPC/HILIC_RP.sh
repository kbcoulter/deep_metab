#!/bin/bash

echo "Beginning Setup for Both HILIC and RP"
cd ..
git submodule init
git submodule update

apptainer build graphormercontainer.sif docker://dnhem/proj_deepmetab:v0.1 ### NOTE VERSION

mkdir -p graphormer_checkpoints_HILIC
wget -O graphormer_checkpoints_HILIC/99.pt "https://zenodo.org/records/15021743/files/99.pt?download=1" ### EDIT FOR SMALLER
mkdir -p predictions_HILIC
echo "HILIC Setup Complete. Beginning RP."

mkdir -p graphormer_checkpoints_RP
wget -O graphormer_checkpoints_RP/oct30_RP_unc.pt https://zenodo.org/records/15021743/files/oct30_RP_unc.pt?download=1
mkdir -p predictions_HILIC
echo "RP Setup Complete."

echo "Setup Completed Successfully!"