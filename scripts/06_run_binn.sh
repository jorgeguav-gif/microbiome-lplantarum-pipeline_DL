#!/bin/bash

# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif

set -e
source ~/.bashrc
conda activate bio_pytorch

ANALYSIS_DIR="/scratch/users/sgonzalezh852/seq/Seq_Jorge/analysis"

echo "=========================================================="
echo " STARTING BIOLOGICALLY INFORMED NEURAL NETWORK (BINN)"
echo "=========================================================="

python ${ANALYSIS_DIR}/05_deep_learning/06_metabolic_binn.py

echo "=========================================================="
echo " BINN TRAINING COMPLETED."
echo " Check 05_deep_learning/figures/binn_metabolic_importance.tiff"
echo "=========================================================="
