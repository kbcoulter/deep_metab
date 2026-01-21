#!/bin/bash

VERSION=v0.1
IMAGE=dnhem/proj_deepmetab
RWEIGHTS=https://zenodo.org/records/15021743/files/oct30_RP_unc.pt?download=1

echo "Beginning Setup for RP."

cd ..

git submodule init
git submodule update

echo "Building Apptainer image from ${IMAGE}:${VERSION}"
apptainer build graphormercontainer.sif docker://${IMAGE}:${VERSION}

mkdir -p graphormer_checkpoints_RP
wget -O graphormer_checkpoints_RP/RP_WEIGHTS.pt "${RWEIGHTS}"

mkdir -p predictions_RP

echo "RP Setup Completed Successfully!"