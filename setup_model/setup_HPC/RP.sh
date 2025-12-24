#!/bin/bash

echo "Beginning Setup for RP."

cd ..

git submodule init
git submodule update

apptainer build graphormercontainer.sif docker://dnhem/proj_deepmetab:v0.1 ### NOTE VERSION

mkdir -p graphormer_checkpoints_RP
wget -O graphormer_checkpoints_RP/oct30_RP_unc.pt https://zenodo.org/records/15021743/files/oct30_RP_unc.pt?download=1

mkdir -p predictions_HILIC

echo "Setup Completed Successfully!"