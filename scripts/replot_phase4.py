"""
@author: Jorge Luis Gutiérrez-Ávila
@institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
@orcid: https://orcid.org/0000-0003-1630-954X
@github: jorgeguav-gif
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import SpectralEmbedding
from scipy.stats import rankdata
import os

base_dir = r"./data"
metadata_file = os.path.join(base_dir, 'metadata.csv')
otu_file = os.path.join(base_dir, '03_classification', 'combined', 'otu_table.csv')
figures_dir = os.path.join(base_dir, 'figures')

os.makedirs(figures_dir, exist_ok=True)

meta = pd.read_csv(metadata_file, index_col=0)
otu = pd.read_csv(otu_file, index_col=0)

if len(otu) > 100:
    otu = otu.T

comunes = meta.index.intersection(otu.index)
meta = meta.loc[comunes]
otu = otu.loc[comunes]

otu_pseudo = otu + 0.5
otu_clr = np.log(otu_pseudo).sub(np.log(otu_pseudo).mean(axis=1), axis=0)

se = SpectralEmbedding(n_components=2, affinity='nearest_neighbors', random_state=42)
emb = se.fit_transform(otu_clr)

meta['Dim1'] = emb[:, 0]
meta['Dim2'] = emb[:, 1]

if meta.loc[meta['treatment'] == 'Control', 'Dim1'].mean() > meta.loc[meta['treatment'] != 'Control', 'Dim1'].mean():
    meta['Dim1'] = -meta['Dim1']

meta['Pseudotiempo'] = (rankdata(meta['Dim1']) - 1) / (len(meta) - 1)

plt.style.use('default')
sns.set_theme(style="whitegrid")

palette = {
    'Control': '#3498db',

    'LM20': '#2ecc71',

    'G7': '#f39c12',

    'PS128': '#e74c3c'

}
markers = {
    'Control': 'o',

    'LM20': 's',

    'G7': '^',

    'PS128': 'D'

}

plt.figure(figsize=(10, 8))
for trat in palette.keys():
    subset = meta[meta['treatment'] == trat]
    plt.scatter(subset['Dim1'], subset['Dim2'], 
                color=palette[trat], 
                marker=markers[trat], 
                s=150, 
                label=trat, 
                edgecolors='black', 
                linewidths=1.5,
                alpha=0.85)

x_min, x_max = meta['Dim1'].min(), meta['Dim1'].max()
y_mean_start = meta.loc[meta['Dim1'] < x_min + (x_max - x_min)*0.2, 'Dim2'].mean()
y_mean_end = meta.loc[meta['Dim1'] > x_max - (x_max - x_min)*0.2, 'Dim2'].mean()
plt.annotate('', xy=(x_max, y_mean_end), xytext=(x_min, y_mean_start),
             arrowprops=dict(facecolor='gray', shrink=0.05, alpha=0.3, width=15, headwidth=30))

plt.title('Trayectoria Evolutiva del Ecosistema\n(Spectral Embedding)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Dimensión Latente 1 (Eje Evolutivo)', fontsize=14)
plt.ylabel('Dimensión Latente 2', fontsize=14)
plt.legend(title='Treatment', fontsize=12, title_fontsize=14, loc='best')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'trayectoria_2d_ecosistema.svg'), dpi=300)
plt.savefig(os.path.join(figures_dir, 'trayectoria_2d_ecosistema.tiff'), dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
ax = sns.violinplot(x='treatment', y='Pseudotiempo', data=meta, palette=palette, inner='quartile', linewidth=2)
sns.stripplot(x='treatment', y='Pseudotiempo', data=meta, color='black', size=6, alpha=0.7, jitter=True, marker='o')

plt.title('Maduración del Ecosistema Intestinal:\nProgresión del Pseudotiempo por Treatment', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Treatment Probiótico', fontsize=14)
plt.ylabel('Pseudotiempo Biológico (0 = Disbiosis, 1 = Clímax)', fontsize=14)
plt.ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'pseudotiempo_distribucion.svg'), dpi=300)
plt.savefig(os.path.join(figures_dir, 'pseudotiempo_distribucion.tiff'), dpi=300)
plt.close()

print("Graficas de la Phase 4 regeneradas con nuevos colores y marcadores.")
