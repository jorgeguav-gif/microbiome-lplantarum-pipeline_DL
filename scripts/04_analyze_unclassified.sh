#!/bin/bash

# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif

set -e

echo "=================================================================="
echo "  IDENTIFICACIÓN DE 'MATERIA OSCURA' (UNCLASSIFIED) VÍA BLAST"
echo "=================================================================="

source ~/.bashrc
conda activate bio_pytorch

LOTE1_UNCLASS_DIR="/scratch/users/sgonzalezh852/seq/Seq_Jorge/instances/wf-16s_01KY5ZYFHPBEC3QHHEWDRVFEKM_L1/output/unclassified"
ANALYSIS_DIR="/scratch/users/sgonzalezh852/seq/Seq_Jorge/analysis"

FASTQ_FILE=$(find "${LOTE1_UNCLASS_DIR}" -name "*.fastq.gz" | head -n 1)

if [ -z "${FASTQ_FILE}" ]; then
    echo "ERROR: Not found ningún file fastq.gz en ${LOTE1_UNCLASS_DIR}"
    exit 1
fi

echo "File Unclassified detectado: ${FASTQ_FILE}"
echo "Starting BLAST remoto (puede demorar unos minutos)..."

python "${ANALYSIS_DIR}/hpc_scripts/04_blast_unclassified.py" \
    --fastq "${FASTQ_FILE}" \
    --output "${ANALYSIS_DIR}/04_statistics/results/unclassified_blast_report.csv" \
    --n_seqs 30

echo "¡Completed! Revisa el file unclassified_blast_report.csv"
