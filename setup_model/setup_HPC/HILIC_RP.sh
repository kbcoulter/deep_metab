#!/bin/bash

VERSION=v0.1
IMAGE=dnhem/proj_deepmetab
HWEIGHTS=https://zenodo.org/records/15021743/files/99.pt?download=1
RWEIGHTS=https://zenodo.org/records/15021743/files/oct30_RP_unc.pt?download=1

echo "Beginning Setup for Both HILIC and RP"
cd ..
git submodule init
git submodule update

echo "Building Apptainer image from ${IMAGE}:${VERSION}"
apptainer build graphormercontainer.sif docker://${IMAGE}:${VERSION}

mkdir -p graphormer_checkpoints_HILIC
wget -O graphormer_checkpoints_HILIC/HILIC_WEIGHTS.pt "${HWEIGHTS}" ### EDIT FOR SMALLER
mkdir -p predictions_HILIC
echo "HILIC Setup Complete. Beginning RP."

mkdir -p graphormer_checkpoints_RP
wget -O graphormer_checkpoints_RP/RP_WEIGHTS.pt "${RWEIGHTS}"
mkdir -p predictions_RP
echo "RP Setup Complete."

echo "Setup Completed Successfully!"