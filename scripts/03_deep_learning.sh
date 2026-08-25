#!/bin/bash

# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif

set -e

echo "=================================================================="
echo "  PIPELINE DE DEEP LEARNING - MICROBIOMA 16S"
echo "=================================================================="
echo "Fecha de inicio   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "ID del trabajo    : ${SLURM_JOB_ID}"
echo "Nombre del trabajo: ${SLURM_JOB_NAME}"
echo "Nodo asignado     : ${SLURM_NODELIST}"
echo "Allocated CPUs    : ${SLURM_CPUS_PER_TASK}"
echo "Allocated Memory  : ${SLURM_MEM_PER_NODE} MB"
echo "Allocated GPUs    : ${SLURM_GPUS_ON_NODE:-1}"
echo "Working Directory: $(pwd)"
echo "=================================================================="

SECONDS=0

ANALYSIS_DIR="/scratch/users/sgonzalezh852/seq/Seq_Jorge/analysis"
DL_DIR="${ANALYSIS_DIR}/05_deep_learning"
RESULTS_DIR="${DL_DIR}/results"
MODELS_DIR="${DL_DIR}/models"
FIGURES_DIR="${DL_DIR}/figures"

echo ""
echo ">>> [1/6] Configuring directories..."
mkdir -p "${RESULTS_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${FIGURES_DIR}"
mkdir -p "${ANALYSIS_DIR}/logs"

echo "    DL Directory      : ${DL_DIR}"
echo "    Models Directory : ${MODELS_DIR}"
echo "    Figures Directory : ${FIGURES_DIR}"

echo ""
echo ">>> [2/6] Configuring Python + PyTorch environment..."

if command -v module &> /dev/null; then
    echo "    Loading system modules..."
    module purge
    module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true
    module load cudnn 2>/dev/null || true
fi

echo "    Activating conda environment..."
source ~/.bashrc
conda activate bio_pytorch 2>/dev/null || {
    echo "WARNING: Could not activate conda environment."
    echo "Trying with base environment..."
    conda activate base 2>/dev/null || true
}

echo "    Python: $(which python)"
echo "    Python Version: $(python --version 2>&1)"

echo ""
echo ">>> [3/6] Configuring CUDA and GPU..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0

echo "    Checking GPU availability..."
python -c "
import torch
print(f'    PyTorch version   : {torch.__version__}')
print(f'    CUDA available   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    GPU device   : {torch.cuda.get_device_name(0)}')
    print(f'    Total GPU memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'    CUDA version      : {torch.version.cuda}')
    print(f'    cuDNN version     : {torch.backends.cudnn.version()}')
else:
    print('    WARNING: GPU not available, using CPU')
" || {
    echo "    ERROR: PyTorch is not installed or could not be imported"
    echo "    Install with: pip install torch torchvision"
    exit 1
}

echo ""
echo ">>> [4/6] Checking Python dependencies..."
python -c "
paquetes = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'sklearn': 'Scikit-learn',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
}
faltantes = []
for pkg, nombre in paquetes.items():
    try:
        __import__(pkg)
        print(f'    ✓ {nombre}')
    except ImportError:
        print(f'    ✗ {nombre} - NOT INSTALLED')
        faltantes.append(pkg)

if faltantes:
    print(f'\n    Missing packages: {faltantes}')
    print('    Install with: pip install', ' '.join(faltantes))
" || echo "    Error verificando dependencias"

echo ""
echo ">>> [5/6] Running Deep Learning pipeline..."

DL_SCRIPT="${DL_DIR}/microbiome_dl_pipeline.py"
OTU_TABLE="${ANALYSIS_DIR}/03_classification/combined/otu_table.csv"
METADATA="${ANALYSIS_DIR}/metadata.csv"

if [ ! -f "${DL_SCRIPT}" ]; then
    echo "ERROR: Script not found: ${DL_SCRIPT}"
    exit 1
fi

echo ""
echo "  ═══ PROJECT A: PHENOTYPE PREDICTOR (Multi-head MLP) ═══"
echo "  Start: $(date '+%H:%M:%S')"

python "${DL_SCRIPT}" predict \
    --otu_table "${OTU_TABLE}" \
    --metadata "${METADATA}" \
    --output_dir "${RESULTS_DIR}/proyecto_a" \
    --models_dir "${MODELS_DIR}" \
    --figures_dir "${FIGURES_DIR}" \
    --epochs 200 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --cv_folds 8 \
    --seed 42

echo "  Project A End: $(date '+%H:%M:%S')"

echo ""
echo "  ═══ PROJECT B: VAE FOR EUBIOSIS ═══"
echo "  Start: $(date '+%H:%M:%S')"

python "${DL_SCRIPT}" vae \
    --otu_table "${OTU_TABLE}" \
    --metadata "${METADATA}" \
    --output_dir "${RESULTS_DIR}/proyecto_b" \
    --models_dir "${MODELS_DIR}" \
    --figures_dir "${FIGURES_DIR}" \
    --latent_dim 2 \
    --epochs 300 \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --seed 42

echo "  Project B End: $(date '+%H:%M:%S')"

echo ""
echo ">>> [6/6] Checking generated results..."

for proyecto in proyecto_a proyecto_b; do
    dir="${RESULTS_DIR}/${proyecto}"
    if [ -d "${dir}" ]; then
        n_files=$(find "${dir}" -type f | wc -l)
        echo "  ✓ ${proyecto}: ${n_files} files generated"
    else
        echo "  ✗ ${proyecto}: directory not found"
    fi
done

N_MODELS=$(find "${MODELS_DIR}" -type f -name "*.pt" -o -name "*.pth" 2>/dev/null | wc -l)
N_FIGURES=$(find "${FIGURES_DIR}" -type f -name "*.tiff" 2>/dev/null | wc -l)

echo ""
echo "  Saved models: ${N_MODELS}"
echo "  Generated figures: ${N_FIGURES}"

echo ""
echo "  Final GPU usage:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"

ELAPSED=$SECONDS
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECS=$((ELAPSED % 60))

echo ""
echo "=================================================================="
echo "  DEEP LEARNING PIPELINE COMPLETED"
echo "=================================================================="
echo "End Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total time         : ${HOURS}h ${MINUTES}m ${SECS}s"
echo "Executed projects : A (Predictor) y B (VAE)"
echo "Exit code     : 0"
echo "=================================================================="
