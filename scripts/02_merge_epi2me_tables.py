"""
# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif
"""

#!/usr/bin/env python3
"""
Script para unificar las tablas de abundancia de EPI2ME (wf-16s) de los 3 Lotes.
Este script:
1. Busca los archivos de abundancia generados por EPI2ME para Lote1, Lote2 y Lote3.
2. Renombra las columnas de barcodes para que coincidan con la metadata (ej. barcode01 en Lote1 -> L1_BC01).
3. Transpone la matriz para que las Filas sean Muestras y las Columnas sean Taxas (Requisito para Machine Learning).
4. Une los tres lotes rellenando con 0 las especies que no aparezcan en alguno de los lotes.
5. Exporta el resultado final a analysis/03_classification/combined/otu_table.csv
"""

import os
import glob
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CLASS_DIR = SCRIPT_DIR.parent / "03_classification"
OUT_DIR = CLASS_DIR / "combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOTE_PREFIXES = {
    "Lote1": "L1",
    "Lote2": "L2",
    "Lote3": "L3"
}

dfs_lotes = []

print("==========================================================")
print("Iniciando unificación de tablas de abundancia de EPI2ME...")
print("==========================================================")

for lote, prefix in LOTE_PREFIXES.items():
    lote_dir = CLASS_DIR / f"epi2me_{lote}"
    
    if not lote_dir.exists():
        print(f"[Advertencia] El directorio {lote_dir} no existe aún. ¿Ya terminó EPI2ME para este lote?")
        continue
    
    archivos_tsv = glob.glob(str(lote_dir / "**" / "*abundance*.tsv"), recursive=True)
    archivos_csv = glob.glob(str(lote_dir / "**" / "*abundance*.csv"), recursive=True)
    
    archivos = archivos_tsv + archivos_csv
    
    if not archivos:
        archivos = glob.glob(str(lote_dir / "**" / "taxonomic_report*.*"), recursive=True)
            
    if not archivos:
        print(f"[Error] No se encontró ninguna tabla de abundancia en {lote_dir}")
        continue
        
    target_file = archivos[0]
    print(f"[{lote}] Procesando: {target_file}")
    
    if target_file.endswith('.tsv'):
        df = pd.read_csv(target_file, sep='\t')
    else:
        df = pd.read_csv(target_file)
    
    taxa_col = None
    for col in df.columns:
        if col.lower() in ['name', 'taxname', 'taxon', 'scientific_name', 'taxonomy', 'tax']:
            taxa_col = col
            break
            
    if not taxa_col:
        print(f"  -> No se pudo identificar la columna taxonómica en {lote}. Omitiendo.")
        continue
        
    barcode_cols = [c for c in df.columns if 'barcode' in c.lower() or 'l' in c.lower()]
    if not barcode_cols:
        barcode_cols = [c for c in df.columns if c != taxa_col] # Si no dicen barcode, tomar el resto
        
    df_abund = df[[taxa_col] + barcode_cols].copy()
    
    df_abund = df_abund.dropna(subset=[taxa_col])
    
    df_abund = df_abund.groupby(taxa_col).sum().reset_index()
    
    # 5. Transponer para que Filas = Muestras, Columnas = Taxas
    df_abund.set_index(taxa_col, inplace=True)
    df_t = df_abund.transpose()

    meta_df = pd.read_csv(CLASS_DIR.parent / "metadata.csv")
    
    nuevos_indices = {}
    for idx in df_t.index:
        import re
        numeros = re.findall(r'\d+', idx)
        if numeros:
            num = str(numeros[-1]).zfill(2)
            bc_str = f"barcode{num}"
            
            match = meta_df[(meta_df['lote'] == lote) & (meta_df['barcode'] == bc_str)]
            if not match.empty:
                nuevos_indices[idx] = match['sample_id'].values[0]
            else:
                nuevos_indices[idx] = f"{prefix}_{idx}" # Fallback
        else:
            nuevos_indices[idx] = f"{prefix}_{idx}"
            
    df_t.rename(index=nuevos_indices, inplace=True)
    dfs_lotes.append(df_t)

if dfs_lotes:
    print("\n[INFO] Uniendo matrices de los tres lotes...")
    df_final = pd.concat(dfs_lotes, axis=0, join='outer')
    df_final.fillna(0, inplace=True)
    
    meta_df = pd.read_csv(CLASS_DIR.parent / "metadata.csv")
    muestras_validas = meta_df['sample_id'].dropna().tolist()
    df_final = df_final[df_final.index.isin(muestras_validas)]
    
    out_file = OUT_DIR / "otu_table.csv"
    df_final.index.name = "sample_id"
    df_final.to_csv(out_file)
    
    print("==========================================================")
    print(f"¡Éxito! Tabla unificada guardada en: {out_file}")
    print(f"Dimensiones finales: {df_final.shape[0]} muestras, {df_final.shape[1]} especies taxonómicas.")
    print("El proyecto ahora está completamente listo para el script de Deep Learning.")
    print("==========================================================")
else:
    print("No se pudieron combinar las tablas. Revise los mensajes de error.")
