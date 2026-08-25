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
echo "CPUs asignados    : ${SLURM_CPUS_PER_TASK}"
echo "Memoria asignada  : ${SLURM_MEM_PER_NODE} MB"
echo "GPUs asignadas    : ${SLURM_GPUS_ON_NODE:-1}"
echo "Directory trabajo: $(pwd)"
echo "=================================================================="

SECONDS=0

ANALYSIS_DIR="/scratch/users/sgonzalezh852/seq/Seq_Jorge/analysis"
DL_DIR="${ANALYSIS_DIR}/05_deep_learning"
RESULTS_DIR="${DL_DIR}/results"
MODELS_DIR="${DL_DIR}/models"
FIGURES_DIR="${DL_DIR}/figures"

echo ""
echo ">>> [1/6] Configurando directorios..."
mkdir -p "${RESULTS_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${FIGURES_DIR}"
mkdir -p "${ANALYSIS_DIR}/logs"

echo "    Directory DL      : ${DL_DIR}"
echo "    Directory modelos : ${MODELS_DIR}"
echo "    Directory figuras : ${FIGURES_DIR}"

echo ""
echo ">>> [2/6] Configurando entorno Python + PyTorch..."

if command -v module &> /dev/null; then
    echo "    Loading módulos del sistema..."
    module purge
    module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true
    module load cudnn 2>/dev/null || true
fi

echo "    Activando entorno conda..."
source ~/.bashrc
conda activate bio_pytorch 2>/dev/null || {
    echo "ADVERTENCIA: No se pudo activar entorno conda."
    echo "Intentando con el entorno base..."
    conda activate base 2>/dev/null || true
}

echo "    Python: $(which python)"
echo "    Versión Python: $(python --version 2>&1)"

echo ""
echo ">>> [3/6] Configurando CUDA y GPU..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0

echo "    Verificando disponibilidad de GPU..."
python -c "
import torch
print(f'    PyTorch versión   : {torch.__version__}')
print(f'    CUDA disponible   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    GPU dispositivo   : {torch.cuda.get_device_name(0)}')
    print(f'    Memoria GPU total : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'    CUDA versión      : {torch.version.cuda}')
    print(f'    cuDNN versión     : {torch.backends.cudnn.version()}')
else:
    print('    ADVERTENCIA: GPU no disponible, se usará CPU')
" || {
    echo "    ERROR: PyTorch no está instalado o no se pudo importar"
    echo "    Instale con: pip install torch torchvision"
    exit 1
}

echo ""
echo ">>> [4/6] Verificando dependencias de Python..."
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
        print(f'    ✗ {nombre} - NO INSTALADO')
        faltantes.append(pkg)

if faltantes:
    print(f'\n    Paquetes faltantes: {faltantes}')
    print('    Instale con: pip install', ' '.join(faltantes))
" || echo "    Error verificando dependencias"

echo ""
echo ">>> [5/6] Ejecutando pipeline de Deep Learning..."

DL_SCRIPT="${DL_DIR}/microbiome_dl_pipeline.py"
OTU_TABLE="${ANALYSIS_DIR}/03_classification/combined/otu_table.csv"
METADATA="${ANALYSIS_DIR}/metadata.csv"

if [ ! -f "${DL_SCRIPT}" ]; then
    echo "ERROR: Script no encontrado: ${DL_SCRIPT}"
    exit 1
fi

echo ""
echo "  ═══ PROYECTO A: PREDICTOR DE FENOTIPO (Multi-head MLP) ═══"
echo "  Inicio: $(date '+%H:%M:%S')"

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

echo "  Fin Proyecto A: $(date '+%H:%M:%S')"

echo ""
echo "  ═══ PROYECTO B: VAE PARA EUBIOSIS ═══"
echo "  Inicio: $(date '+%H:%M:%S')"

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

echo "  Fin Proyecto B: $(date '+%H:%M:%S')"


echo ""
echo ">>> [6/6] Verificando results generados..."

for proyecto in proyecto_a proyecto_b; do
    dir="${RESULTS_DIR}/${proyecto}"
    if [ -d "${dir}" ]; then
        n_files=$(find "${dir}" -type f | wc -l)
        echo "  ✓ ${proyecto}: ${n_files} files generados"
    else
        echo "  ✗ ${proyecto}: directory no encontrado"
    fi
done

N_MODELS=$(find "${MODELS_DIR}" -type f -name "*.pt" -o -name "*.pth" 2>/dev/null | wc -l)
N_FIGURES=$(find "${FIGURES_DIR}" -type f -name "*.tiff" 2>/dev/null | wc -l)

echo ""
echo "  Modelos guardados: ${N_MODELS}"
echo "  Figuras generadas: ${N_FIGURES}"

echo ""
echo "  Uso final de GPU:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi no disponible)"

ELAPSED=$SECONDS
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECS=$((ELAPSED % 60))

echo ""
echo "=================================================================="
echo "  PIPELINE DE DEEP LEARNING COMPLETADO"
echo "=================================================================="
echo "Fecha de finalización: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Time total         : ${HOURS}h ${MINUTES}m ${SECS}s"
echo "Proyectos ejecutados : A (Predictor) y B (VAE)"
echo "Código de salida     : 0"
echo "=================================================================="
