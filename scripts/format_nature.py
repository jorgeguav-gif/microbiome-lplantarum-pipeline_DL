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
    'Sex Prediction': 'Sex Prediction',
    'Treatment Prediction': 'Treatment Prediction',
    'Predicho': 'Predicted',
    'Real': 'Actual',
    'Importance (Gradient Magnitude)': 'Importance (Gradient Magnitude)',
    'Top 20 Most Important Taxa for Prediction': 'Top 20 Most Important Taxa for Prediction',
    'Reconstruction of Control vs Treated': 'Reconstruction of Control vs Treated',
    'Reconstruction Error (MSE)': 'Reconstruction Error (MSE)',
    'Reconstruction Error Distribution': 'Reconstruction Error Distribution',
    'Latent Space (PCA)': 'Latent Space (PCA)',
    'Principal Component 1': 'Principal Component 1',
    'Principal Component 2': 'Principal Component 2',
    'Confusion Matrix': 'Confusion Matrix',
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
