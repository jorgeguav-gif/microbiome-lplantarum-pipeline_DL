#!/bin/bash

# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif

set -eo pipefail

echo "=================================================================="
echo "  16S MICROBIOME STATISTICAL ANALYSIS"
echo "=================================================================="
echo "Start date   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Assigned node     : ${SLURM_NODELIST}"
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
echo "    Analysis directory : ${ANALYSIS_DIR}"
echo "    Statistics directory: ${STATS_DIR}"
echo "    Results directory  : ${RESULTS_DIR}"
echo "    Figures Directory     : ${FIGURES_DIR}"

echo ""
echo ">>> [2/5] Configuring Python environment..."
source ~/.bashrc
conda activate bio_pytorch 2>/dev/null || {
    echo "ERROR: Could not load bio_pytorch environment."
    exit 1
}

echo "    Python found at: $(which python)"
echo "    Python version   : $(python --version 2>&1)"

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
    echo "    ✓ Python script completed successfully"
    
    N_RESULTS=$(find "${RESULTS_DIR}" -type f 2>/dev/null | wc -l)
    N_FIGURES=$(find "${FIGURES_DIR}" -type f -name "*.tiff" 2>/dev/null | wc -l)
    
    echo "    ✓ Generated results files: ${N_RESULTS}"
    echo "    ✓ Generated figures: ${N_FIGURES}"
    
    echo ""
    echo "    Results:"
    ls -lh "${RESULTS_DIR}/" 2>/dev/null || echo "    (empty)"
    echo ""
    echo "    Figures:"
    ls -lh "${FIGURES_DIR}/" 2>/dev/null || echo "    (empty)"
else
    echo "    ✗ ERROR: Script failed with exit code ${RSCRIPT_EXIT}"
    echo "    Check error file: logs/stats_${SLURM_JOB_ID}.err"
    exit ${RSCRIPT_EXIT}
fi

ELAPSED=$SECONDS
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECS=$((ELAPSED % 60))

echo ""
echo "=================================================================="
echo "  STATISTICAL ANALYSIS COMPLETED"
echo "=================================================================="
echo "End Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total time         : ${HOURS}h ${MINUTES}m ${SECS}s"
echo "Exit code     : 0"
echo "=================================================================="
