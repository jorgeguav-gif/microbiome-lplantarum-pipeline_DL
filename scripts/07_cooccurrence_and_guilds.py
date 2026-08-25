"""
# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from networkx.algorithms import community
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# CONFIGURACIÓN ESTÉTICA (NATURE STYLE)
# ==========================================
sns.set_style('ticks')
plt.rcParams.update({'font.family': 'sans-serif', 'savefig.dpi': 300, 'pdf.fonttype': 42})

def clean_tax_name(idx):
    """Limpia el nombre taxonómico de EMU (toma nivel de Género o Especie)"""
    parts = str(idx).split(';')
    if len(parts) > 0:
        return parts[-1].strip()
    return str(idx)

def fdr_bh(pvals):
    """Corrección de Benjamini-Hochberg (Tasa de Falso Descubrimiento)"""
    pvals = np.asarray(pvals)
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    fdr = np.zeros(n)
    for i, p in enumerate(sorted_pvals):
        fdr[i] = p * n / (i + 1)
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    fdr = np.minimum(fdr, 1.0)
    
    out_fdr = np.zeros(n)
    out_fdr[sorted_idx] = fdr
    return out_fdr

# ==========================================
# 1. ENFOQUE TRADICIONAL: RED DE COOCURRENCIA CLR
# ==========================================
def build_cooccurrence_network(clr_df, out_dir):
    print("\n[INFO] Construyendo Red de Coocurrencia Ecológica (Tradicional)...")
    
    taxa = clr_df.columns.tolist()
    n_taxa = len(taxa)
    
    # Matrices de correlación y p-valores
    corr_mat = np.zeros((n_taxa, n_taxa))
    pval_mat = np.zeros((n_taxa, n_taxa))
    
    print("[INFO] Calculando coeficiente de Spearman sobre espacio CLR...")
    for i in range(n_taxa):
        for j in range(i, n_taxa):
            if i == j:
                corr_mat[i, j] = 1.0
                pval_mat[i, j] = 0.0
            else:
                rho, p = spearmanr(clr_df.iloc[:, i], clr_df.iloc[:, j])
                corr_mat[i, j] = rho
                corr_mat[j, i] = rho
                pval_mat[i, j] = p
                pval_mat[j, i] = p
                
    # Extraer p-valores del triángulo superior para corrección FDR
    upper_tri_idx = np.triu_indices(n_taxa, k=1)
    pvals_flat = pval_mat[upper_tri_idx]
    fdr_flat = fdr_bh(pvals_flat)
    
    # Reconstruir matriz FDR
    fdr_mat = np.zeros((n_taxa, n_taxa))
    fdr_mat[upper_tri_idx] = fdr_flat
    fdr_mat.T[upper_tri_idx] = fdr_flat
    
    # Crear el grafo NetworkX
    G = nx.Graph()
    for t in taxa:
        G.add_node(t)
        
    edges_added = 0
    # Criterio ecológico: |Rho| > 0.30 y p-valor < 0.05
    for i in range(n_taxa):
        for j in range(i + 1, n_taxa):
            if abs(corr_mat[i, j]) > 0.30 and pval_mat[i, j] < 0.05:
                G.add_edge(taxa[i], taxa[j], weight=corr_mat[i, j])
                edges_added += 1
                
    print(f"[INFO] Red construida: {n_taxa} nodos y {edges_added} interacciones significativas.")
    
    # Remover nodos desconectados (huérfanos)
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    if len(G.nodes) == 0:
        print("[WARNING] Ninguna interacción pasó los filtros estadísticos estrictos.")
        return
        
    # Detectar Gremios (Comunidades) usando Louvain/Modularity
    print("[INFO] Detectando Gremios Funcionales usando Modularity (Louvain alternativo)...")
    communities = list(community.greedy_modularity_communities(G))
    
    # Asignar color a cada comunidad
    color_map = []
    palette = sns.color_palette("Set2", len(communities))
    for node in G:
        for i, comm in enumerate(communities):
            if node in comm:
                color_map.append(palette[i])
                break
                
    # Asignar color de línea según si la interacción es sinérgica (+) o antagonista (-)
    edge_colors = ['#d73027' if G[u][v]['weight'] < 0 else '#4575b4' for u, v in G.edges()]
    
    # Dibujar la Red
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=color_map, alpha=0.9, edgecolors='white', linewidths=0.5)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.6, width=1.0)
    
    # Etiquetas para todos los nodos
    labels = {n: n for n in G.nodes()}
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='sans-serif', font_weight='bold')
    
    # Crear Leyenda para los Gremios
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=palette[i], label=f'Gremio {i+1} ({len(comm)} taxones)') for i, comm in enumerate(communities)]
    plt.legend(handles=legend_handles, loc='upper left', title="Gremios Funcionales", bbox_to_anchor=(1.05, 1))
    
    plt.title("Red de Coocurrencia Ecológica y Gremios Funcionales (Algoritmo Louvain)", fontsize=14, pad=20)
    plt.axis('off')
    
    out_path_tiff = os.path.join(out_dir, "red_coocurrencia_tradicional.tiff")
    plt.savefig(out_path_tiff, format='tiff', dpi=300, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
    
    out_path_svg = os.path.join(out_dir, "red_coocurrencia_tradicional.svg")
    plt.savefig(out_path_svg, format='svg', bbox_inches='tight')
    
    plt.close()
    print(f"[EXITO] Gráficas guardadas en: {out_dir} (TIFF y SVG)")


# ==========================================
# 2. ENFOQUE DEEP LEARNING: SPARSE AUTOENCODER (SAE)
# ==========================================
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SparseAutoencoder, self).__init__()
        # Encoder: Comprimir el microbioma en Gremios Latentes
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.relu = nn.ReLU()
        # Decoder: Intentar reconstruir el microbioma original
        self.decoder = nn.Linear(latent_dim, input_dim)
        
    def forward(self, x):
        encoded = self.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_autoencoder_and_extract_guilds(clr_df, out_dir, meta):
    print("\n[INFO] Entrenando Autoencoder Disperso (Deep Learning) para encontrar Gremios Latentes...")
    
    taxa = clr_df.columns.tolist()
    X = torch.tensor(clr_df.values, dtype=torch.float32)
    
    input_dim = X.shape[1]
    latent_dim = 10 # Forzamos a la red a agrupar todas las bacterias en 10 dimensiones ecológicas máximas
    
    model = SparseAutoencoder(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    mse_loss = nn.MSELoss()
    
    # L1 Regularization factor (induce parsimonia/dispersión)
    lambda_l1 = 0.001
    
    epochs = 500
    for epoch in range(epochs):
        optimizer.zero_grad()
        encoded, decoded = model(X)
        
        # Loss = Error de Reconstrucción + Penalización L1 sobre los pesos del Encoder
        loss_recon = mse_loss(decoded, X)
        l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() if 'encoder.weight' in name)
        
        loss = loss_recon + lambda_l1 * l1_norm
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f"       Epoch {epoch+1}/{epochs} | Loss Total: {loss.item():.4f} (Recon: {loss_recon.item():.4f})")
            
    # Extraer los pesos sinápticos del Encoder
    # Shape = [latent_dim, input_dim] -> [10, N_Bacterias]
    encoder_weights = model.encoder.weight.detach().numpy()
    
    # Transponemos para que las filas sean Bacterias y columnas sus "Pesos Latentes"
    weights_df = pd.DataFrame(encoder_weights.T, index=taxa, columns=[f"Latent_{i+1}" for i in range(latent_dim)])
    
    # Calcular distancias y realizar clustering jerárquico
    print("[INFO] Realizando Clustering Jerárquico sobre el Espacio Latente...")
    dist_matrix = pdist(weights_df.values, metric='cosine')
    linkage_matrix = linkage(dist_matrix, method='ward')
    
    # Graficar el Clustermap (Heatmap con Dendrograma)
    plt.figure(figsize=(10, 12))
    
    # Solo graficamos las 60 bacterias con mayor señal absoluta para que sea legible
    importance = np.abs(weights_df.values).sum(axis=1)
    top_indices = np.argsort(importance)[-60:]
    top_taxa = weights_df.iloc[top_indices]
    
    g = sns.clustermap(top_taxa, metric='euclidean', method='ward', cmap='RdBu_r', 
                       figsize=(12, 10), standard_scale=1, center=0,
                       cbar_kws={'label': 'Intensidad Sináptica (Escalada)'})
    
    g.fig.suptitle("Gremios Funcionales Latentes descubiertos por el Autoencoder Disperso", y=1.02, fontsize=14)
    g.ax_heatmap.set_ylabel("Taxones Microbianos")
    g.ax_heatmap.set_xlabel("Vectores Latentes (Gremios)")
    
    out_path_tiff = os.path.join(out_dir, "gremios_latentes_deep_learning.tiff")
    g.savefig(out_path_tiff, format='tiff', dpi=300, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
    
    out_path_svg = os.path.join(out_dir, "gremios_latentes_deep_learning.svg")
    g.savefig(out_path_svg, format='svg', bbox_inches='tight')
    
    plt.close()
    print(f"[EXITO] Gráficas estructurales guardadas en: {out_dir} (TIFF y SVG)")
    
    # ========================================================
    # 3. EXTRAER ACTIVACIONES POR MUESTRA Y CRUZAR CON METADATA
    # ========================================================
    print("[INFO] Generando Heatmap de Activaciones por Tratamiento y Sexo...")
    model.eval()
    with torch.no_grad():
        activations, _ = model(X) # (n_samples, latent_dim)
        
    act_df = pd.DataFrame(activations.numpy(), index=clr_df.index, columns=[f"Gremio_{i+1}" for i in range(latent_dim)])
    
    common_idx = act_df.index.intersection(meta.index)
    if len(common_idx) > 0:
        act_df = act_df.loc[common_idx]
        act_df['Tratamiento'] = meta.loc[common_idx, 'tratamiento']
        act_df['Sexo'] = meta.loc[common_idx, 'sexo']
        
        # Promediar por Tratamiento y Sexo
        act_mean = act_df.groupby(['Tratamiento', 'Sexo']).mean()
        
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(act_mean, cmap='viridis', annot=True, fmt=".2f", linewidths=.5, cbar_kws={'label': 'Nivel de Activación Medio'})
        plt.title("Nivel de Activación de Gremios Funcionales por Cohorte Biológica", pad=20, fontsize=14)
        plt.ylabel("Cohorte (Tratamiento, Sexo)", fontsize=12)
        plt.xlabel("Gremios Latentes", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        out_act_tiff = os.path.join(out_dir, "activacion_gremios_tratamiento.tiff")
        plt.savefig(out_act_tiff, format='tiff', dpi=300, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
        out_act_svg = os.path.join(out_dir, "activacion_gremios_tratamiento.svg")
        plt.savefig(out_act_svg, format='svg', bbox_inches='tight')
        plt.close()
        print(f"[EXITO] Heatmap de Activaciones guardado en {out_dir}")
    else:
        print("[ADVERTENCIA] No hubo cruce entre IDs de muestra y metadata. Imposible graficar activaciones.")



# ==========================================
# FLUJO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    base_dir = "/scratch/users/sgonzalezh852/seq/Seq_Jorge/analysis"
    otu_path = os.path.join(base_dir, "03_classification", "combined", "otu_table.csv")
    meta_path = os.path.join(base_dir, "metadata.csv")
    out_dir = os.path.join(base_dir, "05_deep_learning", "figures")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Cargar datos
    print("[INFO] Cargando tabla de OTUs y Metadata...")
    meta = pd.read_csv(meta_path, index_col='sample_id')
    otu = pd.read_csv(otu_path, index_col=0) # Filas: Muestras, Columnas: Especies
    if 'Unclassified' in otu.columns:
        otu = otu.drop('Unclassified', axis=1)
        
    # Limpiar nombres para las gráficas
    otu.columns = [clean_tax_name(idx) for idx in otu.columns]
    
    # Agrupar columnas duplicadas si limpiar nombres generó colisiones
    otu = otu.T.groupby(otu.columns).sum().T
    
    # 2. Filtrado Ecológico y Transformación CLR
    print("[INFO] Filtrando taxones de baja prevalencia y aplicando transformación CLR...")
    # Mantener bacterias presentes en al menos el 20% de las muestras (ruido)
    prevalence = (otu > 0).mean(axis=0)
    otu_filtered = otu.loc[:, prevalence > 0.20]
    
    # Manejo de Ceros (Pseudocount de 0.5 a la abundancia relativa)
    # Primero convertimos a cuentas relativas si no lo están, asumimos que son abundancias relativas (0-1 o 0-100)
    otu_pseudo = otu_filtered + 0.5 
    
    # Transformación CLR (Centered Log-Ratio) manual
    # CLR(x) = ln(x) - mean(ln(x))
    log_data = np.log(otu_pseudo)
    clr_data = log_data.subtract(log_data.mean(axis=1), axis=0)
    
    # 3. Ejecutar Enfoque Tradicional
    build_cooccurrence_network(clr_data, out_dir)
    
    # Entrenar Autoencoder para encontrar la estructura de Gremios + Activaciones
    train_autoencoder_and_extract_guilds(clr_data, out_dir, meta)
    
    print("\n[INFO] PIPELINE DE COOCURRENCIA Y GREMIOS COMPLETADO EXITOSAMENTE.")
