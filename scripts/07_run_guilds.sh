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
echo " INICIANDO ANÁLISIS DUAL DE COOCURRENCIA Y GREMIOS"
echo " (Tradicional CLR-Spearman + Deep Learning Autoencoder)"
echo "=========================================================="

python ${ANALYSIS_DIR}/05_deep_learning/07_cooccurrence_and_guilds.py

echo "=========================================================="
echo " ANÁLISIS COMPLETADO."
echo " Revisa 05_deep_learning/figures/red_coocurrencia_tradicional.tiff"
echo " y gremios_latentes_deep_learning.tiff"
echo "=========================================================="
