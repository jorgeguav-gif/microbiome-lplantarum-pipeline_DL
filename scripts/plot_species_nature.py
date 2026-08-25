"""
@author: Jorge Luis Gutiérrez-Ávila
@institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
@orcid: https://orcid.org/0000-0003-1630-954X
@github: jorgeguav-gif
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# =============================================================================
nature_colors = [
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', 
    '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85',
    '#FFB547', '#9370DB', '#20B2AA', '#FF69B4', '#CD5C5C', '#4682B4'
]
sns.set_palette(sns.color_palette(nature_colors))
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
    'legend.fontsize': 7,
    'figure.titlesize': 10,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8
})

def save_nature_tiff(fig, filepath):
    """Guarda la figura en TIFF con compresión LZW y 300 DPI"""
    fig.savefig(filepath, dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'}, bbox_inches='tight')
    fig.savefig(filepath.replace('.tiff', '.svg'), format='svg', bbox_inches='tight')

def clean_tax_name(tax_string):
    """Limpia el string taxonómico de EPI2ME para extraer solo la especie"""
    if "Unclassified" in str(tax_string):
        return "Unclassified"
    parts = str(tax_string).split(';')
    if len(parts) > 0:
        return parts[-1].strip()
    return tax_string

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir

    otu_path = os.path.join(base_dir, "03_classification", "combined", "otu_table.csv")
    meta_path = os.path.join(base_dir, "metadata.csv")
    out_dir = os.path.join(base_dir, "04_statistics", "figures")

    os.makedirs(out_dir, exist_ok=True)

    print("[INFO] Cargando tabla maestra de especies (OTU table) y metadatos...")
    otu = pd.read_csv(otu_path, index_col=0).T
    
    otu.index = [clean_tax_name(idx) for idx in otu.index]
    
    meta = pd.read_csv(meta_path, index_col='sample_id')
    
    samples_comunes = otu.columns.intersection(meta.index)
    otu = otu[samples_comunes]
    meta = meta.loc[samples_comunes]
    
    if 'Unclassified' in otu.index:
        print("[INFO] Eliminando grupo 'Unclassified' del análisis...")
        otu = otu.drop('Unclassified')

    print("[INFO] Convirtiendo a abundancias relativas (solo especies conocidas)...")
    otu_rel = otu.div(otu.sum(axis=0), axis=1) * 100  # Porcentaje

    top_n = 15
    top_species = otu_rel.mean(axis=1).nlargest(top_n).index
    
    otu_top = otu_rel.loc[top_species].copy()
    otu_top.loc['Others'] = 100 - otu_top.sum(axis=0)

    df_plot = otu_top.T.join(meta[['tratamiento']])
    df_grouped = df_plot.groupby('tratamiento').mean()

    # =========================================================================
    # =========================================================================
    print("[INFO] Exportando la composición exacta del grupo 'Others'...")
    others_species = otu_rel.index[~otu_rel.index.isin(top_species)]
    df_others = otu_rel.loc[others_species].mean(axis=1).sort_values(ascending=False).to_frame(name='Abundancia_Media_General_Porcentaje')
    df_others.index.name = 'Taxon'
    df_others.to_csv(os.path.join(out_dir, "others_composition.csv"))

    # =========================================================================
    # =========================================================================
    print("[INFO] Generating Stacked Bar Plot (Nature Style)...")
    fig, ax = plt.subplots(figsize=(6, 5))
    
    bottom = np.zeros(len(df_grouped))
    for i, species in enumerate(df_grouped.columns):
        color = nature_colors[i % len(nature_colors)] if species != 'Others' else '#D3D3D3'
        
        label_sp = species
        if species not in ['Others', 'Unclassified']:
            label_sp = f"$\\mathit{{{species.replace('_', ' ')}وة}}$".replace('وة', '')
            
        ax.bar(df_grouped.index, df_grouped[species], bottom=bottom, label=label_sp, color=color, width=0.7)
        bottom += df_grouped[species]

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    ax.set_ylabel("Relative Abundance (%)", fontweight='bold')
    ax.set_xlabel("")
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)
    
    save_nature_tiff(fig, os.path.join(out_dir, "species_abundance_bars_en.tiff"))
    
    ax.set_ylabel("Abundancia Relativa (%)", fontweight='bold')
    save_nature_tiff(fig, os.path.join(out_dir, "species_abundance_bars_es.tiff"))
    plt.close(fig)

    # =========================================================================
    # =========================================================================
    print("[INFO] Generating Heatmap de Especies...")
    
    top_20 = otu_rel.mean(axis=1).nlargest(20).index
    df_heat_full = otu_rel.loc[top_20].T.join(meta[['tratamiento']])
    df_heat = df_heat_full.groupby('tratamiento').mean().T
    
    df_heat_z = df_heat.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    italic_idx = []
    for sp in df_heat_z.index:
        if sp not in ['Others', 'Unclassified']:
            italic_idx.append(f"$\\mathit{{{sp.replace('_', ' ')}وة}}$".replace('وة', ''))
        else:
            italic_idx.append(sp)
    df_heat_z.index = italic_idx

    cmap_npg = LinearSegmentedColormap.from_list("NPG_heat", ["#3C5488", "white", "#E64B35"])
    
    fig = plt.figure(figsize=(7, 6))
    g = sns.clustermap(
        df_heat_z, 
        cmap=cmap_npg, 
        col_cluster=False, 
        row_cluster=True,
        linewidths=0.5, 
        linecolor='white',
        cbar_kws={'label': 'Z-score (Abundance)'},
        figsize=(6, 6)
    )
    
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.set_xlabel("Treatment")
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    
    save_nature_tiff(g.fig, os.path.join(out_dir, "species_heatmap_en.tiff"))
    
    g.cax.set_ylabel('Z-score (Abundancia)')
    g.ax_heatmap.set_xlabel("Tratamiento")
    save_nature_tiff(g.fig, os.path.join(out_dir, "species_heatmap_es.tiff"))
    plt.close('all')

    print("[INFO] ¡Plots de especies generados exitosamente en la carpeta 'figures'!")

if __name__ == "__main__":
    main()
