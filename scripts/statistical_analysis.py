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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
from sklearn.decomposition import PCA

def shannon_index(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def simpson_index(counts):
    p = counts / counts.sum()
    return 1 - np.sum(p**2)

def chao1_index(counts):
    s_obs = (counts > 0).sum()
    singletons = (counts == 1).sum()
    doubletons = (counts == 2).sum()
    if doubletons > 0:
        return s_obs + (singletons**2) / (2 * doubletons)
    else:
        return s_obs + (singletons * (singletons - 1)) / 2

def dominance_index(counts):
    """
    Dominance Index (Simpson's dominance) = sum(p_i^2)
    It ranges from 0 to 1, where larger values mean one or few taxa dominate the community.
    Reference: Simpson, E.H. (1949). Measurement of diversity. Nature, 163(4148), 688.
    """
    p = counts / counts.sum()
    return np.sum(p**2)

def clr_transform(counts, pseudocount=1.0):
    counts_pseudo = counts + pseudocount
    log_counts = np.log(counts_pseudo)
    return log_counts.subtract(log_counts.mean(axis=1), axis=0)

def save_tiff(fig, filepath):
    fig.savefig(filepath, dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'}, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser(description="Microbiome Statistics (Aitchison/CLR) in Python")
    parser.add_argument("--otu_table", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--figures_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print("[INFO] Cargando datos...")
    otu = pd.read_csv(args.otu_table, index_col=0)
    meta = pd.read_csv(args.metadata, index_col='sample_id')
    
    samples_comunes = otu.index.intersection(meta.index)
    otu = otu.loc[samples_comunes]
    meta = meta.loc[samples_comunes]

    print("[INFO] Calculating Diversidad Alfa...")
    alpha_div = pd.DataFrame(index=otu.index)
    alpha_div['Shannon'] = otu.apply(shannon_index, axis=1)
    alpha_div['Simpson'] = otu.apply(simpson_index, axis=1)
    alpha_div['Chao1'] = otu.apply(chao1_index, axis=1)
    alpha_div['Dominance'] = otu.apply(dominance_index, axis=1)
    
    df_alpha = alpha_div.join(meta)
    df_alpha.to_csv(os.path.join(args.output_dir, "alpha_diversity.csv"))
    
    metrics = [
        ('Shannon', 'Shannon Diversity Index', 'Índice de Diversidad Shannon'),
        ('Simpson', 'Simpson Diversity Index', 'Índice de Diversidad Simpson'),
        ('Chao1', 'Chao1 Richness Estimator', 'Estimador de Riqueza Chao1'),
        ('Dominance', 'Dominance Index (Simpson)', 'Índice de Dominancia (Simpson)')
    ]

    for metric, title_en, title_es in metrics:
        fig = plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_alpha, x='tratamiento', y=metric, color='lightblue')
        plt.title(title_en)
        plt.xlabel('Treatment')
        plt.ylabel(metric)
        save_tiff(fig, os.path.join(args.figures_dir, f"alpha_{metric.lower()}_en.tiff"))
        
        plt.title(title_es)
        plt.xlabel('Tratamiento')
        plt.ylabel(metric)
        save_tiff(fig, os.path.join(args.figures_dir, f"alpha_{metric.lower()}_es.tiff"))
        plt.close(fig)

        fig = plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_alpha, x='tratamiento', y=metric, hue='sexo')
        plt.title(f"{title_en} by Sex")
        plt.xlabel('Treatment')
        plt.ylabel(metric)
        plt.legend(title='Sex')
        save_tiff(fig, os.path.join(args.figures_dir, f"alpha_{metric.lower()}_sex_en.tiff"))
        
        plt.title(f"{title_es} por Sexo")
        plt.xlabel('Tratamiento')
        plt.ylabel(metric)
        plt.legend(title='Sexo')
        save_tiff(fig, os.path.join(args.figures_dir, f"alpha_{metric.lower()}_sex_es.tiff"))
        plt.close(fig)

    print("[INFO] Aplicando transformación CLR (Aitchison)...")
    otu_clr = clr_transform(otu, pseudocount=0.5)
    
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(otu_clr)
    var_exp = pca.explained_variance_ratio_ * 100
    
    df_beta = pd.DataFrame(coords, columns=['PC1', 'PC2'], index=otu.index).join(meta)
    df_beta.to_csv(os.path.join(args.output_dir, "beta_diversity_aitchison.csv"))
    
    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_beta, x='PC1', y='PC2', hue='tratamiento', style='sexo', s=100)
    plt.title(f"Aitchison PCA by Treatment\nPC1 ({var_exp[0]:.1f}%) - PC2 ({var_exp[1]:.1f}%)")
    plt.xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    plt.ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    save_tiff(fig, os.path.join(args.figures_dir, "beta_aitchison_tratamiento_en.tiff"))

    plt.title(f"PCA de Aitchison por Tratamiento\nPC1 ({var_exp[0]:.1f}%) - PC2 ({var_exp[1]:.1f}%)")
    save_tiff(fig, os.path.join(args.figures_dir, "beta_aitchison_tratamiento_es.tiff"))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_beta, x='PC1', y='PC2', hue='batch', s=100, palette='Set2')
    plt.title(f"Batch Effect Evaluation (Aitchison PCA)\nPC1 ({var_exp[0]:.1f}%) - PC2 ({var_exp[1]:.1f}%)")
    plt.xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    plt.ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    save_tiff(fig, os.path.join(args.figures_dir, "beta_aitchison_batch_en.tiff"))

    plt.title(f"Evaluación de Efecto de Batch (PCA de Aitchison)\nPC1 ({var_exp[0]:.1f}%) - PC2 ({var_exp[1]:.1f}%)")
    save_tiff(fig, os.path.join(args.figures_dir, "beta_aitchison_batch_es.tiff"))
    plt.close(fig)

    print("[INFO] Corrigiendo efecto de batch (Mean-centering en CLR)...")
    otu_clr_corrected = otu_clr.copy()
    global_means = otu_clr.mean(axis=0)
    for batch in meta['batch'].unique():
        batch_idx = meta['batch'] == batch
        batch_means = otu_clr[batch_idx].mean(axis=0)
        otu_clr_corrected[batch_idx] = otu_clr[batch_idx] - batch_means + global_means
    
    pca_corr = PCA(n_components=2, random_state=42)
    coords_corr = pca_corr.fit_transform(otu_clr_corrected)
    var_exp_corr = pca_corr.explained_variance_ratio_ * 100
    
    df_beta_corr = pd.DataFrame(coords_corr, columns=['PC1', 'PC2'], index=otu.index).join(meta)
    df_beta_corr.to_csv(os.path.join(args.output_dir, "beta_diversity_aitchison_corrected.csv"))
    
    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_beta_corr, x='PC1', y='PC2', hue='batch', s=100, palette='Set2')
    plt.title(f"Batch Effect Corrected (Aitchison PCA)\nPC1 ({var_exp_corr[0]:.1f}%) - PC2 ({var_exp_corr[1]:.1f}%)")
    plt.xlabel(f"PC1 ({var_exp_corr[0]:.1f}%)")
    plt.ylabel(f"PC2 ({var_exp_corr[1]:.1f}%)")
    save_tiff(fig, os.path.join(args.figures_dir, "beta_aitchison_batch_corrected_en.tiff"))

    plt.title(f"Efecto de Batch Corregido (PCA de Aitchison)\nPC1 ({var_exp_corr[0]:.1f}%) - PC2 ({var_exp_corr[1]:.1f}%)")
    save_tiff(fig, os.path.join(args.figures_dir, "beta_aitchison_batch_corrected_es.tiff"))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_beta_corr, x='PC1', y='PC2', hue='tratamiento', style='sexo', s=100)
    plt.title(f"Aitchison PCA by Treatment (Batch-Corrected)\nPC1 ({var_exp_corr[0]:.1f}%) - PC2 ({var_exp_corr[1]:.1f}%)")
    plt.xlabel(f"PC1 ({var_exp_corr[0]:.1f}%)")
    plt.ylabel(f"PC2 ({var_exp_corr[1]:.1f}%)")
    save_tiff(fig, os.path.join(args.figures_dir, "beta_aitchison_tratamiento_corrected_en.tiff"))

    plt.title(f"PCA de Aitchison por Tratamiento (Corregido por Batch)\nPC1 ({var_exp_corr[0]:.1f}%) - PC2 ({var_exp_corr[1]:.1f}%)")
    save_tiff(fig, os.path.join(args.figures_dir, "beta_aitchison_tratamiento_corrected_es.tiff"))
    plt.close(fig)

    otu_clr = otu_clr_corrected

    print("[INFO] Realizando análisis diferencial top 20 taxa...")
    otu_rel = otu.div(otu.sum(axis=1), axis=0) 
    top_taxa = otu_rel.mean().nlargest(20).index
    resultados_kw = []
    
    for taxa in top_taxa:
        grupos = [otu_clr[taxa][meta['tratamiento'] == t].values for t in meta['tratamiento'].unique()]
        try:
            stat, p_val = kruskal(*grupos)
            resultados_kw.append({'Taxon': taxa, 'H_stat': stat, 'p_value': p_val})
        except:
            pass
            
    df_kw = pd.DataFrame(resultados_kw).sort_values('p_value')
    df_kw.to_csv(os.path.join(args.output_dir, "diferencial_kruskal_wallis_clr.csv"), index=False)

    print("[INFO] ¡Análisis composicional completed exitosamente!")

if __name__ == "__main__":
    main()
