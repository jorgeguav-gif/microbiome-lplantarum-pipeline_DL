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

nature_setup = 

content = re.sub(
    r'import matplotlib.*?import seaborn as sns', 
    nature_setup, 
    content, 
    flags=re.DOTALL
)

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

content = re.sub(
    r'plt\.savefig\((.*?), dpi=300\)',
    r"plt.savefig(\1, dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})",
    content
)
content = content.replace('.png', '.tiff')

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
