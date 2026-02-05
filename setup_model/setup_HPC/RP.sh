#!/bin/bash
set -e

# Default
VERSION_DEFAULT=v0.3
IMAGE_DEFAULT=dnhem/proj_deepmetab
RWEIGHTS_DEFAULT=https://zenodo.org/records/15021743/files/oct30_RP_unc.pt?download=1

# Active
VERSION="${VERSION_DEFAULT}"
IMAGE="${IMAGE_DEFAULT}"
RWEIGHTS="${RWEIGHTS_DEFAULT}"

# Args
require_arg() {
  if [[ -z "$2" || "$2" == --* ]]; then
    echo "Error: $1 requires a value"
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version-default)
      VERSION="${VERSION_DEFAULT}"
      shift
      ;;
    --version-custom)
      require_arg "$1" "$2"
      VERSION="$2"
      shift 2
      ;;
    --image-default)
      IMAGE="${IMAGE_DEFAULT}"
      shift
      ;;
    --image-custom)
      require_arg "$1" "$2"
      IMAGE="$2"
      shift 2
      ;;
    --rweights-default)
      RWEIGHTS="${RWEIGHTS_DEFAULT}"
      shift
      ;;
    --rweights-custom)
      require_arg "$1" "$2"
      RWEIGHTS="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage:"
      echo "  $0 [options]"
      echo ""
      echo "Version:"
      echo "  --version-default"
      echo "  --version-custom VALUE"
      echo ""
      echo "Image:"
      echo "  --image-default"
      echo "  --image-custom VALUE"
      echo ""
      echo "RP Weights:"
      echo "  --rweights-default"
      echo "  --rweights-custom URL"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

#Setup
echo "Beginning Setup for RP."
cd ../..
git submodule init
git submodule update
echo "Building Apptainer image from ${IMAGE}:${VERSION}"
apptainer build graphormercontainer.sif docker://${IMAGE}:${VERSION}
mkdir -p graphormer_checkpoints_RP
wget -O graphormer_checkpoints_RP/RP_WEIGHTS.pt "${RWEIGHTS}"
mkdir -p predictions_RP
echo "Setup Completed With Config:"
echo "  IMAGE    = ${IMAGE}"
echo "  VERSION  = ${VERSION}"
echo "  RWEIGHTS = ${RWEIGHTS}"