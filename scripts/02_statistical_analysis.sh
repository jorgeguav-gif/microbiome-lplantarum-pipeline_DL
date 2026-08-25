#!/bin/bash

# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif

set -eo pipefail

echo "=================================================================="
echo "  ANÁLISIS ESTADÍSTICO DEL MICROBIOMA 16S"
echo "=================================================================="
echo "Fecha de inicio   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "ID del trabajo    : ${SLURM_JOB_ID}"
echo "Nombre del trabajo: ${SLURM_JOB_NAME}"
echo "Nodo asignado     : ${SLURM_NODELIST}"
echo "Allocated CPUs : ${SLURM_CPUS_PER_TASK}"
echo "Allocated Memory  : ${SLURM_MEM_PER_NODE} MB"
echo "Working Directory: $(pwd)"
echo "=================================================================="

SECONDS=0

ANALYSIS_DIR="/scratch/users/sgonzalezh852/seq/Seq_Jorge/analysis"
STATS_DIR="${ANALYSIS_DIR}/04_statistics"
RESULTS_DIR="${STATS_DIR}/results"
FIGURES_DIR="${STATS_DIR}/figures"

echo ""
echo ">>> [1/5] Configuring directories..."
mkdir -p "${RESULTS_DIR}"
mkdir -p "${FIGURES_DIR}"
mkdir -p "${ANALYSIS_DIR}/logs"
echo "    Directory de análisis : ${ANALYSIS_DIR}"
echo "    Directory estadísticas: ${STATS_DIR}"
echo "    Directory results  : ${RESULTS_DIR}"
echo "    Figures Directory     : ${FIGURES_DIR}"

echo ""
echo ">>> [2/5] Configurando entorno Python..."
source ~/.bashrc
conda activate bio_pytorch 2>/dev/null || {
    echo "ERROR: No se pudo cargar el entorno bio_pytorch."
    exit 1
}

echo "    Python found en: $(which python)"
echo "    Versión de Python   : $(python --version 2>&1)"

echo ""
echo ">>> [3/5] Running statistical analysis in Python..."
echo "    Start: $(date '+%H:%M:%S')"

python "${STATS_DIR}/statistical_analysis.py" \
    --otu_table "${ANALYSIS_DIR}/03_classification/combined/otu_table.csv" \
    --metadata "${ANALYSIS_DIR}/metadata.csv" \
    --output_dir "${RESULTS_DIR}" \
    --figures_dir "${FIGURES_DIR}"

RSCRIPT_EXIT=$?

echo "    End: $(date '+%H:%M:%S')"

echo ""
echo ">>> [5/5] Checking generated results..."

if [ ${RSCRIPT_EXIT} -eq 0 ]; then
    echo "    ✓ Script R completed exitosamente"
    
    N_RESULTS=$(find "${RESULTS_DIR}" -type f 2>/dev/null | wc -l)
    N_FIGURES=$(find "${FIGURES_DIR}" -type f -name "*.tiff" 2>/dev/null | wc -l)
    
    echo "    ✓ Files de results generados: ${N_RESULTS}"
    echo "    ✓ Generated figures: ${N_FIGURES}"
    
    echo ""
    echo "    Results:"
    ls -lh "${RESULTS_DIR}/" 2>/dev/null || echo "    (vacío)"
    echo ""
    echo "    Figuras:"
    ls -lh "${FIGURES_DIR}/" 2>/dev/null || echo "    (vacío)"
else
    echo "    ✗ ERROR: El script R falló con código de salida ${RSCRIPT_EXIT}"
    echo "    Revise el file de error: logs/stats_${SLURM_JOB_ID}.err"
    exit ${RSCRIPT_EXIT}
fi

ELAPSED=$SECONDS
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECS=$((ELAPSED % 60))

echo ""
echo "=================================================================="
echo "  ANÁLISIS ESTADÍSTICO COMPLETED"
echo "=================================================================="
echo "End Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total time         : ${HOURS}h ${MINUTES}m ${SECS}s"
echo "Exit code     : 0"
echo "=================================================================="
