#!/bin/bash

# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif

set -e

echo "=========================================================="
echo "Starting EPI2ME wf-16s pipeline"
echo "SLURM Job: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Date: $(date)"
echo "=========================================================="

source ~/.bashrc
conda activate nextflow_env

if [ -z "$1" ]; then
    echo "Error: You must specify the batch. Example: sbatch 01b_epi2me_wf16s.sh Batch1"
    exit 1
fi

LOTE=$1

BASE_DIR="/scratch/users/sgonzalezh852/seq/Seq_Jorge"

RUN_DIR="${BASE_DIR}/${LOTE}" 
CORRIDA=$(ls -d ${RUN_DIR}/* | head -n 1)
FASTQ_DIR="${CORRIDA}/fastq_pass"

OUT_DIR="${BASE_DIR}/analysis/03_classification/epi2me_${LOTE}"
NEXTFLOW_CACHE="${BASE_DIR}/.nextflow_cache"

mkdir -p $OUT_DIR
mkdir -p $NEXTFLOW_CACHE
export NXF_SINGULARITY_CACHEDIR=$NEXTFLOW_CACHE

echo "Analyzing Batch: $LOTE"
echo "FASTQ Directory: $FASTQ_DIR"

cd $OUT_DIR

mkdir -p "$CONDA_PREFIX/var/singularity/mnt/session"

mkdir -p "$OUT_DIR/bin"
cat << 'EOF_WRAP' > "$OUT_DIR/bin/singularity"
#!/bin/bash
REAL_SING="REPLACE_ME_SINGULARITY_PATH"
if [ ! -x "$REAL_SING" ]; then
    echo "Error: Singularity not found at $REAL_SING"
    exit 1
fi
args=()
for arg in "$@"; do
    if [ "$arg" != "--no-home" ]; then
        args+=("$arg")
    fi
done
exec "$REAL_SING" "${args[@]}"
EOF_WRAP

sed -i "s|REPLACE_ME_SINGULARITY_PATH|$CONDA_PREFIX/bin/singularity|g" "$OUT_DIR/bin/singularity"
chmod +x "$OUT_DIR/bin/singularity"
export PATH="$OUT_DIR/bin:$PATH"

nextflow run epi2me-labs/wf-16s \
    -revision master \
    -profile singularity \
    --fastq "$FASTQ_DIR" \
    --out_dir "$OUT_DIR" \
    --taxonomic_rank S \
    --threads 8

echo "=========================================================="
echo "Pipeline finished: $(date)"
echo "Results at: $OUT_DIR"
echo "=========================================================="
