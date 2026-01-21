#!/bin/bash

VERSION=v0.1
IMAGE=dnhem/proj_deepmetab
HWEIGHTS=https://zenodo.org/records/15021743/files/99.pt?download=1

echo "Beginning Setup for HILIC."

cd ..

git submodule init
git submodule update

echo "Building Apptainer image from ${IMAGE}:${VERSION}"
apptainer build graphormercontainer.sif docker://${IMAGE}:${VERSION}

mkdir -p graphormer_checkpoints_HILIC
wget -O graphormer_checkpoints_HILIC/HILIC_WEIGHTS.pt "${HWEIGHTS}"

mkdir -p predictions_HILIC

echo "HILIC Setup Completed Successfully!"