#!/bin/bash

# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif

set -eo pipefail

echo "=================================================================="
echo "  STARTING PHASE 2: PhILR (Python)"
echo "=================================================================="
echo "Start date        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

source ~/.bashrc
conda activate bio_pytorch 2>/dev/null || conda activate microbiota_env

BASE_DIR="/scratch/users/sgonzalezh852/seq/Seq_Jorge/analysis"
PY_SCRIPT="${BASE_DIR}/04_statistics/philr_analysis.py"

echo "Running PhILR analysis script in Python..."
python "${PY_SCRIPT}" \
    --otu_table "${BASE_DIR}/03_classification/combined/otu_table.csv" \
    --metadata "${BASE_DIR}/metadata.csv" \
    --output_dir "${BASE_DIR}/04_statistics/results" \
    --figures_dir "${BASE_DIR}/figures"

echo "=================================================================="
echo "  PHASE 2 COMPLETED"
echo "=================================================================="
echo "End date          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="
