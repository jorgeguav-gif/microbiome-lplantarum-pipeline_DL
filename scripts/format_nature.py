"""
@author: Jorge Luis Gutiérrez-Ávila
@institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
@orcid: https://orcid.org/0000-0003-1630-954X
@github: jorgeguav-gif
"""

import re
import sys

file_path = r'C:\Users\Lenovo\Documents\seq\Seq_Jorge\analysis\05_deep_learning\microbiome_dl_pipeline.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Configuración global de Nature y Matplotlib
nature_setup = """import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para HPC
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración estilo Nature
nature_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
sns.set_palette(sns.color_palette(nature_colors))
sns.set_context('paper', font_scale=1.2)
sns.set_style('ticks')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 10
})"""

content = re.sub(
    r'import matplotlib.*?import seaborn as sns', 
    nature_setup, 
    content, 
    flags=re.DOTALL
)

# 2. Traducciones al Inglés Británico
replacements = {
    'Predicción de Sexo': 'Sex Prediction',
    'Predicción de Tratamiento': 'Treatment Prediction',
    'Predicho': 'Predicted',
    'Real': 'Actual',
    'Importancia (magnitud del gradiente)': 'Importance (Gradient Magnitude)',
    'Top 20 Taxa más Importantes para Predicción': 'Top 20 Most Important Taxa for Prediction',
    'Reconstrucción de Control vs Tratados': 'Reconstruction of Control vs Treated',
    'Error de Reconstrucción (MSE)': 'Reconstruction Error (MSE)',
    'Distribución del Error de Reconstrucción': 'Reconstruction Error Distribution',
    'Espacio Latente (PCA)': 'Latent Space (PCA)',
    'Componente Principal 1': 'Principal Component 1',
    'Componente Principal 2': 'Principal Component 2',
    'Matriz de Confusión': 'Confusion Matrix',
    'Precisión': 'Accuracy'
}

for es, en in replacements.items():
    content = content.replace(es, en)

# 3. Guardar como TIFF con LZW compresión
content = re.sub(
    r'plt\.savefig\((.*?), dpi=300\)',
    r"plt.savefig(\1, dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})",
    content
)
content = content.replace('.png', '.tiff')

# 4. Convertir las etiquetas del eje Y (taxa) en formato cursiva (italics)
# Expresión original: ax.set_yticklabels([taxa_nombres[i] for i in top_idx][::-1], fontsize=8)
# Nueva expresión: ax.set_yticklabels([f"$\mathit{{{taxa_nombres[i].replace('_', ' ')}}}$" for i in top_idx][::-1], fontsize=8)

content = re.sub(
    r'ax\.set_yticklabels\(\[taxa_nombres\[i\] for i in top_idx\]\[::-1\], fontsize=8\)',
    r'ax.set_yticklabels([f"$\\mathit{{{taxa_nombres[i].replace(\'_\', \' \')}}}$" for i in top_idx][::-1], fontsize=8)',
    content
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Modificaciones completadas correctamente.")
