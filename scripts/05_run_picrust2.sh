#!/bin/bash

# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif

set -e
source ~/.bashrc

ANALYSIS_DIR="/scratch/users/sgonzalezh852/seq/Seq_Jorge/analysis"
FASTA="${ANALYSIS_DIR}/04_statistics/picrust2_input.fasta"
TSV="${ANALYSIS_DIR}/04_statistics/picrust2_input.tsv"
OUT_DIR="${ANALYSIS_DIR}/04_statistics/picrust2_out"

echo "=========================================================="
echo "STEP 1: Download 16S sequences from NCBI (Biopython)"
echo "=========================================================="
conda activate bio_pytorch
python ${ANALYSIS_DIR}/hpc_scripts/05_prepare_picrust2.py

echo "=========================================================="
echo " STEP 2: Run PICRUSt2"
echo "=========================================================="

echo "Cleaning previous installation..."
rm -rf ${ANALYSIS_DIR}/picrust2_env

cd ${ANALYSIS_DIR}/hpc_scripts
if [ ! -f "micromamba" ] && [ ! -f "picrust2.sif" ]; then
    wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
    mv bin/micromamba .
    rm -rf bin
fi

if [ ! -f "picrust2.sif" ]; then
    echo "Installing PICRUSt2 v2.5.3 with Micromamba (including 'r' channel)..."
    ./micromamba create -y -p ${ANALYSIS_DIR}/picrust2_env -c conda-forge -c bioconda -c r -c defaults picrust2=2.5.3 || {
        echo "Micromamba failed due to cluster libraries. Activating Plan B: Containers..."
        if command -v apptainer &> /dev/null; then
            echo "Using Apptainer to download the container..."
            apptainer pull picrust2.sif docker://picrust/picrust2:2.5.2
        elif command -v singularity &> /dev/null; then
            echo "Using Singularity to download the container..."
            singularity pull picrust2.sif docker://picrust/picrust2:2.5.2
        else
            echo "CRITICAL: Could not install via Conda and Singularity/Apptainer are not available on this cluster."
            exit 1
        fi
    }
fi

rm -rf ${OUT_DIR}

if [ -f "picrust2.sif" ]; then
    echo "Running via isolated container..."
    if command -v apptainer &> /dev/null; then
        apptainer exec picrust2.sif picrust2_pipeline.py -s ${FASTA} -i ${TSV} -o ${OUT_DIR} -p 8 --verbose
    else
        singularity exec picrust2.sif picrust2_pipeline.py -s ${FASTA} -i ${TSV} -o ${OUT_DIR} -p 8 --verbose
    fi
else
    echo "Running via Micromamba..."
    eval "$(./micromamba shell hook -s bash)"
    micromamba activate ${ANALYSIS_DIR}/picrust2_env
    picrust2_pipeline.py -s ${FASTA} -i ${TSV} -o ${OUT_DIR} -p 8 --verbose
fi

echo "=========================================================="
echo " PICRUSt2 Completed. Results at ${OUT_DIR}"
echo "=========================================================="
