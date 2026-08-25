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
echo " PASO 1: Descargar secuencias 16S de NCBI (Biopython)"
echo "=========================================================="
conda activate bio_pytorch
python ${ANALYSIS_DIR}/hpc_scripts/05_prepare_picrust2.py

echo "=========================================================="
echo " PASO 2: Ejecutar PICRUSt2"
echo "=========================================================="

echo "Limpiando instalación anterior..."
rm -rf ${ANALYSIS_DIR}/picrust2_env

cd ${ANALYSIS_DIR}/hpc_scripts
if [ ! -f "micromamba" ] && [ ! -f "picrust2.sif" ]; then
    wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
    mv bin/micromamba .
    rm -rf bin
fi

if [ ! -f "picrust2.sif" ]; then
    echo "Instalando PICRUSt2 v2.5.3 con Micromamba (incluyendo canal 'r')..."
    ./micromamba create -y -p ${ANALYSIS_DIR}/picrust2_env -c conda-forge -c bioconda -c r -c defaults picrust2=2.5.3 || {
        echo "Micromamba falló debido a librerías de tu clúster. Activando Plan B: Contenedores..."
        if command -v apptainer &> /dev/null; then
            echo "Usando Apptainer para descargar el contenedor..."
            apptainer pull picrust2.sif docker://picrust/picrust2:2.5.2
        elif command -v singularity &> /dev/null; then
            echo "Usando Singularity para descargar el contenedor..."
            singularity pull picrust2.sif docker://picrust/picrust2:2.5.2
        else
            echo "CRÍTICO: No se pudo instalar por Conda y Singularity/Apptainer no están disponibles en este clúster."
            exit 1
        fi
    }
fi

rm -rf ${OUT_DIR}

if [ -f "picrust2.sif" ]; then
    echo "Ejecutando a través de contenedor aislado..."
    if command -v apptainer &> /dev/null; then
        apptainer exec picrust2.sif picrust2_pipeline.py -s ${FASTA} -i ${TSV} -o ${OUT_DIR} -p 8 --verbose
    else
        singularity exec picrust2.sif picrust2_pipeline.py -s ${FASTA} -i ${TSV} -o ${OUT_DIR} -p 8 --verbose
    fi
else
    echo "Ejecutando a través de Micromamba..."
    eval "$(./micromamba shell hook -s bash)"
    micromamba activate ${ANALYSIS_DIR}/picrust2_env
    picrust2_pipeline.py -s ${FASTA} -i ${TSV} -o ${OUT_DIR} -p 8 --verbose
fi

echo "=========================================================="
echo " PICRUSt2 Completed. Results en ${OUT_DIR}"
echo "=========================================================="
