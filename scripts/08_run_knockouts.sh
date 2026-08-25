#!/bin/bash

# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif

echo "=================================================================="
echo " IN SILICO KNOCKOUTS PIPELINE (Deep Learning)"
echo "=================================================================="
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Node  : $(hostname)"
echo "CPU(s): ${SLURM_CPUS_PER_TASK}"
echo ""

echo ">>> [1/2] Initializing environment..."
source ~/.bashrc
conda activate bio_pytorch || conda activate base

PROJECT_DIR="/scratch/users/sgonzalezh852/seq/Seq_Jorge/analysis"
DL_SCRIPT="${PROJECT_DIR}/05_deep_learning/08_in_silico_knockouts.py"

if [ ! -f "${DL_SCRIPT}" ]; then
    echo "[ERROR] Python script not found: ${DL_SCRIPT}"
    exit 1
fi

echo ">>> [2/2] Running Computational Extinction Simulations..."
python "${DL_SCRIPT}"

echo ""
echo "=================================================================="
echo " COMPLETED"
echo "=================================================================="
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
