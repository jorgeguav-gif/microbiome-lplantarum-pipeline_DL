"""
@author: Jorge Luis Gutiérrez-Ávila
@institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
@orcid: https://orcid.org/0000-0003-1630-954X
@github: jorgeguav-gif
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from microbiome_dl_pipeline import MicrobiomeDataset, MicrobiomeVAE

def get_dysbiosis_scores(model, X_tensor):
    
    model.eval()
    with torch.no_grad():
        recon_batch, mu, logvar = model(X_tensor)
        mse = torch.mean((recon_batch - X_tensor)**2, dim=1)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        loss = mse + 0.001 * kld
    return loss.numpy()

def main():
    base_dir = "/scratch/users/sgonzalezh852/seq/Seq_Jorge/analysis"
    otu_path = os.path.join(base_dir, "03_classification", "combined", "otu_table.csv")
    meta_path = os.path.join(base_dir, "metadata.csv")
    model_path = os.path.join(base_dir, "05_deep_learning", "models", "vae_eubiosis.pt")
    out_dir = os.path.join(base_dir, "05_deep_learning", "figures")
    os.makedirs(out_dir, exist_ok=True)
    
    print("======================================================")
    print(" IN SILICO KNOCKOUTS: Inferencia de Especies Clave")
    print("======================================================")
    
    print("[INFO] Cargando Dataset...")
    dataset = MicrobiomeDataset(otu_path, meta_path)
    X_rel = dataset.obtener_abundancias(normalizar=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_rel)
    X_tensor = torch.FloatTensor(X_scaled)
    
    taxa = dataset.otu.columns
    input_dim = X_scaled.shape[1]
    
    print("[INFO] Restaurando red neuronal entrenada (VAE)...")
    if not os.path.exists(model_path):
        print(f"[ERROR] Not found el model entrenado en {model_path}")
        print("Asegúrate de haber ejecutado 03_deep_learning.sh previamente.")
        sys.exit(1)
        
    model = MicrobiomeVAE(input_dim=input_dim, latent_dim=2)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    print("[INFO] Calculating puntajes de Eubiosis base (Real)...")
    baseline_scores = get_dysbiosis_scores(model, X_tensor)
    
    print("[INFO] Preparando microcirugía computacional (Knockouts)...")
    mean_abundances = X_rel.mean(axis=0)
    top_indices = np.argsort(mean_abundances)[-50:]
    
    knockout_impacts = []
    
    for i, idx in enumerate(top_indices):
        bacterium = taxa[idx]
        
        X_ko = X_rel.copy()
        X_ko[:, idx] = 0.0 
        
        sumas = X_ko.sum(axis=1, keepdims=True)
        sumas[sumas == 0] = 1 # Evitar división por cero si vaciáramos el microbioma
        X_ko = X_ko / sumas
        
        X_ko_scaled = scaler.transform(X_ko)
        X_ko_tensor = torch.FloatTensor(X_ko_scaled)
        
        ko_scores = get_dysbiosis_scores(model, X_ko_tensor)
        
        # Impacto = Disbiosis(Con Knockout) - Disbiosis(Real)
        # Delta Positivo = La disbiosis empeoró (Extinguir esta bacteria es PELIGROSO)
        # Delta Negativo = La disbiosis mejoró (Esta bacteria era DAÑINA)
        delta = ko_scores - baseline_scores
        mean_impact = np.mean(delta)
        
        simple_name = bacterium.split(';')[-1] if ';' in bacterium else bacterium
        simple_name = simple_name.replace('_', ' ')
        
        knockout_impacts.append({
            'Taxon': simple_name,
            'Impacto_Medio': mean_impact,
            'Impacto_Std': np.std(delta)
        })
        
        if (i+1) % 10 == 0:
            print(f"       {i+1}/50 bacterias procesadas...")
            
    impact_df = pd.DataFrame(knockout_impacts).sort_values(by='Impacto_Medio', ascending=True)
    
    print("[INFO] Generating plot de Causalidad (Impacto Causal)...")
    plt.figure(figsize=(10, 14))
    
    colors = ['#E64B35' if x < 0 else '#4DBBD5' for x in impact_df['Impacto_Medio']]
    
    ax = sns.barplot(data=impact_df, x='Impacto_Medio', y='Taxon', palette=colors)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    plt.title("In Silico Knockouts: Importancia Causal Ecológica (IA)", pad=20, fontsize=15, fontweight='bold')
    plt.xlabel("Impacto en Disbiosis al Extinguir Especie (Δ Error de Reconstrucción)", fontsize=13)
    plt.ylabel("Especie Bacteriana Extinguida", fontsize=13)
    
    plt.text(0.01, 0.98, '← Extinción beneficiosa\n(Patógenos/Disbióticos)', 
             transform=ax.transAxes, color='#E64B35', fontsize=11, fontweight='bold', va='top')
    plt.text(0.99, 0.98, 'Extinción perjudicial →\n(Keystones/Protectores)', 
             transform=ax.transAxes, color='#4DBBD5', fontsize=11, fontweight='bold', ha='right', va='top')
             
    sns.despine()
    plt.tight_layout()
    
    out_tiff = os.path.join(out_dir, "in_silico_knockout_impact.tiff")
    out_svg = os.path.join(out_dir, "in_silico_knockout_impact.svg")
    
    plt.savefig(out_tiff, format='tiff', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(out_svg, format='svg')
    plt.close()
    
    print(f"[EXITO] Análisis finalizado. Plots en {out_dir}")

if __name__ == '__main__':
    main()
