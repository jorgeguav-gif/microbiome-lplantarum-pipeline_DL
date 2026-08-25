#!/bin/bash

# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif

set -eo pipefail

echo "=========================================================="
echo "Starting Phase 3: Metabolic Modeling with MICOM (AGORA1.03)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Fecha y Hora de Start: $(date)"
echo "=========================================================="

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Initializing Conda environment..."
source ~/.bashrc
conda activate microbiota_env

BASE_DIR="/scratch/users/sgonzalezh852/seq/Seq_Jorge/analysis"
MODEL_DIR="${BASE_DIR}/06_metabolic_modeling"
PYTHON_SCRIPT="${MODEL_DIR}/run_micom.py"

cd "${BASE_DIR}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running computational pipeline (64 samples on 32 cores)..."
python "${PYTHON_SCRIPT}"

echo "=========================================================="
echo "Phase 3 Completed Successfully"
echo "End Date and Time: $(date)"
echo "=========================================================="
