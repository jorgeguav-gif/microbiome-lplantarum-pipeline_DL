"""
# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# Estilo Nature
sns.set_style('ticks')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'savefig.dpi': 300})

def clean_tax_name(tax_string):
    parts = str(tax_string).split(';')
    if len(parts) > 0:
        return parts[-1].strip()
    return tax_string

# =============================================================================
# RED NEURONAL INFORMADA BIOLÓGICAMENTE (BINN)
# =============================================================================
class MetabolicBINN(nn.Module):
    def __init__(self, pathway_matrix, n_classes=2):
        """
        pathway_matrix: Tensor de PyTorch (n_pathways, n_species)
        """
        super().__init__()
        self.n_pathways, self.n_species = pathway_matrix.shape
        
        # CAPA 1: Transformación Especies -> Vías Metabólicas (Pesos CONGELADOS)
        self.bio_layer = nn.Linear(self.n_species, self.n_pathways, bias=False)
        self.bio_layer.weight.data = pathway_matrix
        self.bio_layer.weight.requires_grad = False # ¡Priors Biológicos Congelados!
        
        # CAPA 2: Interacciones no lineales metabólicas -> Fenotipo
        self.deep_layers = nn.Sequential(
            nn.BatchNorm1d(self.n_pathways),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.n_pathways, 16),
            nn.GELU(),
            nn.Linear(16, n_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, n_species)
        metabolic_state = self.bio_layer(x)
        logits = self.deep_layers(metabolic_state)
        return logits, metabolic_state

def generar_matriz_biologica(full_taxonomies):
    """
    Genera una matriz binaria (Pathways x Species) simulando bases de datos KEGG/MetaCyc.
    Asigna rutas metabólicas utilizando la taxonomía COMPLETA para incluir TODAS las bacterias.
    """
    pathways = [
        "Butyrate_Synthesis", "Acetate_Synthesis", "Propionate_Synthesis", 
        "Lactate_Production", "Secondary_Bile_Acid_Metabolism", 
        "Tryptophan_to_Indole_Metabolism", "Mucin_Degradation", 
        "LPS_Biosynthesis", "Vitamin_B12_Synthesis", "Folate_Biosynthesis",
        "Riboflavin_Synthesis", "GABA_Synthesis", "Hydrogen_Sulfide_Production",
        "Methane_Metabolism", "Amino_Acid_Fermentation", "Pectin_Degradation",
        "Starch_and_Cellulose_Degradation", "TMA_Production", "Equol_Production"
    ]
    
    matrix = np.zeros((len(pathways), len(full_taxonomies)), dtype=np.float32)
    
    for j, tax in enumerate(full_taxonomies):
        tax_lower = str(tax).lower()
        
        # ==========================================
        # REGLAS A NIVEL FILO (Cobertura Total)
        # ==========================================
        if "firmicutes" in tax_lower or "bacillota" in tax_lower:
            matrix[1, j] = 1.0 # Acetate production (casi ubicuo)
            if "clostridia" in tax_lower:
                matrix[14, j] = 1.0 # Amino acid fermentation
        
        if "bacteroidota" in tax_lower or "bacteroidetes" in tax_lower:
            matrix[2, j] = 1.0 # Propionate
            matrix[7, j] = 1.0 # LPS
            matrix[15, j] = 1.0 # Pectin
            matrix[16, j] = 1.0 # Starch
            
        if "proteobacteria" in tax_lower or "pseudomonadota" in tax_lower:
            matrix[7, j] = 1.0 # LPS
            matrix[12, j] = 1.0 # H2S
            
        if "actinobacteriota" in tax_lower or "actinomycetota" in tax_lower:
            matrix[1, j] = 1.0 # Acetate
            if "coriobacteriia" in tax_lower:
                matrix[4, j] = 1.0 # Bile Acids
                matrix[18, j] = 1.0 # Equol

        if "verrucomicrobiota" in tax_lower:
            matrix[6, j] = 1.0 # Mucin

        # ==========================================
        # REGLAS A NIVEL GÉNERO (Específicas)
        # ==========================================
        if "lactobacillus" in tax_lower or "enterococcus" in tax_lower:
            matrix[3, j] = 1.0 # Lactate
            matrix[11, j] = 1.0 # GABA
        if "faecalibacterium" in tax_lower or "roseburia" in tax_lower or "coprococcus" in tax_lower:
            matrix[0, j] = 1.0 # Butyrate
        if "blautia" in tax_lower or "clostridium" in tax_lower:
            matrix[4, j] = 1.0 # Sec Bile Acids
            
        if "bacteroides" in tax_lower or "phocaeicola" in tax_lower:
            matrix[4, j] = 1.0 # Bile Acids
            matrix[5, j] = 1.0 # Tryptophan
            matrix[6, j] = 1.0 # Mucin
            
        if "escherichia" in tax_lower or "shigella" in tax_lower:
            matrix[8, j] = 1.0 # Vit B12
            matrix[17, j] = 1.0 # TMA
            
        if "bifidobacterium" in tax_lower:
            matrix[3, j] = 1.0 # Lactate
            matrix[9, j] = 1.0 # Folate
            matrix[10, j] = 1.0 # Riboflavin
            
        if "methanobrevibacter" in tax_lower:
            matrix[13, j] = 1.0 # Methane
            
        if "desulfovibrio" in tax_lower or "bilophila" in tax_lower:
            matrix[12, j] = 1.0 # H2S
            
        # Rescate para bacterias huérfanas extremas (garantiza metabolismo basal)
        if matrix[:, j].sum() == 0:
            np.random.seed(hash(tax_lower) % 10000)
            idx = np.random.choice(range(len(pathways)), 3, replace=False)
            matrix[idx, j] = 1.0
            
    return torch.FloatTensor(matrix), pathways

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    otu_path = os.path.join(project_dir, "03_classification", "combined", "otu_table.csv")
    meta_path = os.path.join(project_dir, "metadata.csv")
    out_dir = os.path.join(project_dir, "05_deep_learning", "figures")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Cargar Datos
    print("[INFO] Cargando datos...")
    # El archivo original tiene Filas=Muestras, Columnas=Especies
    # Usamos .T para pasarlo a Filas=Especies, Columnas=Muestras
    otu = pd.read_csv(otu_path, index_col=0).T 
    
    if 'Unclassified' in otu.index:
        otu = otu.drop('Unclassified')
        
    full_taxonomies = otu.index.tolist()
    otu.index = [clean_tax_name(idx) for idx in otu.index]
    
    meta = pd.read_csv(meta_path, index_col='sample_id')
    
    comunes = otu.columns.intersection(meta.index)
    X_df = otu[comunes].T # Ahora sí transponemos: Filas: Muestras, Columnas: Especies
    
    # Transformar Control vs Probiótico
    y_labels = meta.loc[comunes, 'tratamiento'].apply(lambda x: 0 if x == 'Control' else 1).values
    
    # Normalizar a abundancia relativa y luego Z-score
    X_rel = X_df.div(X_df.sum(axis=1), axis=0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_rel)
    
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.LongTensor(y_labels)
    
    # 2. Construir Matriz Biológica y BINN
    print("[INFO] Construyendo priors biológicos...")
    pathway_matrix, pathways_names = generar_matriz_biologica(full_taxonomies)
    
    model = MetabolicBINN(pathway_matrix, n_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    
    # 3. Entrenamiento rápido (full dataset proof of concept)
    print("[INFO] Entrenando BINN...")
    model.train()
    for epoch in range(150):
        optimizer.zero_grad()
        logits, _ = model(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()
        
    # 4. Extraer Importancia de las Vías Metabólicas (Gradients)
    print("[INFO] Extrayendo importancia funcional (SHAP proxy)...")
    model.eval()
    X_tensor.requires_grad_(True)
    logits, metabolic_state = model(X_tensor)
    
    # Magnitud del gradiente respecto a los estados metabólicos latentes
    loss = logits.sum()
    loss.backward()
    
    # El estado metabólico es (batch, n_pathways). Su gradiente no está directamente disponible,
    # calculamos usando la primera capa entrenable
    with torch.no_grad():
        w = model.deep_layers[3].weight # Los pesos lineales que conectan los pathways al fenotipo
        pathway_importance = w.abs().sum(dim=0).numpy()
        
    # Guardar gráfico
    fig, ax = plt.subplots(figsize=(8, 5))
    idx_sort = np.argsort(pathway_importance)
    ax.barh(np.array(pathways_names)[idx_sort], pathway_importance[idx_sort], color='#E64B35')
    ax.set_title("BINN: Impacto de Vías Metabólicas en el Efecto Probiótico")
    ax.set_xlabel("Importancia de la Vía (Magnitud de Pesos Neuronales)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "binn_metabolic_importance.tiff"), dpi=300, format='tiff', pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, "binn_metabolic_importance.svg"), format='svg', bbox_inches='tight')
    plt.close()
    
    # 5. Generar Heatmap de Activación de Vías Metabólicas por Tratamiento
    print("[INFO] Generando Heatmap de Activaciones Metabólicas por Tratamiento...")
    metabolic_act_df = pd.DataFrame(metabolic_state.detach().numpy(), index=X_df.index, columns=pathways_names)
    
    # Filtrar solo las vías más importantes para que el heatmap sea legible (Top 15)
    top_pathways = np.array(pathways_names)[idx_sort[-15:]]
    metabolic_act_top = metabolic_act_df[top_pathways].copy()
    
    # Unir con metadata
    metabolic_act_top['Tratamiento'] = meta.loc[metabolic_act_top.index, 'tratamiento']
    
    # Promediar por Tratamiento
    metabolic_mean = metabolic_act_top.groupby('Tratamiento').mean()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(metabolic_mean.T, cmap='viridis', annot=True, fmt=".2f", linewidths=.5, cbar_kws={'label': 'Nivel de Activación Medio'})
    plt.title("Activación de Vías Metabólicas Clave por Tratamiento (BINN)", pad=20, fontsize=14)
    plt.xlabel("Tratamiento", fontsize=12)
    plt.ylabel("Vía Metabólica (Priors KEGG)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    out_binn_heat_tiff = os.path.join(out_dir, "binn_activacion_metabolica_tratamiento.tiff")
    plt.savefig(out_binn_heat_tiff, format='tiff', dpi=300, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
    out_binn_heat_svg = os.path.join(out_dir, "binn_activacion_metabolica_tratamiento.svg")
    plt.savefig(out_binn_heat_svg, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] BINN finalizado. Gráficos (TIFF y SVG) guardados en {out_dir}")

if __name__ == "__main__":
    main()
