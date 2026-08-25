"""
# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif
"""

#!/usr/bin/env python3

import os
import glob
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CLASS_DIR = SCRIPT_DIR.parent / "03_classification"
OUT_DIR = CLASS_DIR / "combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOTE_PREFIXES = {
    "Batch1": "L1",
    "Batch2": "L2",
    "Batch3": "L3"
}

dfs_batchs = []

print("==========================================================")
print("Starting unificación de tablas de abundancia de EPI2ME...")
print("==========================================================")

for batch, prefix in LOTE_PREFIXES.items():
    batch_dir = CLASS_DIR / f"epi2me_{batch}"
    
    if not batch_dir.exists():
        print(f"[Advertencia] El directorio {batch_dir} no existe aún. ¿Ya terminó EPI2ME para este batch?")
        continue
    
    files_tsv = glob.glob(str(batch_dir / "**" / "*abundance*.tsv"), recursive=True)
    files_csv = glob.glob(str(batch_dir / "**" / "*abundance*.csv"), recursive=True)
    
    files = files_tsv + files_csv
    
    if not files:
        files = glob.glob(str(batch_dir / "**" / "taxonomic_report*.*"), recursive=True)
            
    if not files:
        print(f"[Error] Not found ninguna tabla de abundancia en {batch_dir}")
        continue
        
    target_file = files[0]
    print(f"[{batch}] Processing: {target_file}")
    
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
        print(f"  -> No se pudo identificar la columna taxonómica en {batch}. Omitiendo.")
        continue
        
    barcode_cols = [c for c in df.columns if 'barcode' in c.lower() or 'l' in c.lower()]
    if not barcode_cols:
        barcode_cols = [c for c in df.columns if c != taxa_col] # Si no dicen barcode, tomar el resto
        
    df_abund = df[[taxa_col] + barcode_cols].copy()
    
    df_abund = df_abund.dropna(subset=[taxa_col])
    
    df_abund = df_abund.groupby(taxa_col).sum().reset_index()
    
    # 5. Transponer para que Filas = Samples, Columnas = Taxas
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
            
            match = meta_df[(meta_df['batch'] == batch) & (meta_df['barcode'] == bc_str)]
            if not match.empty:
                nuevos_indices[idx] = match['sample_id'].values[0]
            else:
                nuevos_indices[idx] = f"{prefix}_{idx}" # Fallback
        else:
            nuevos_indices[idx] = f"{prefix}_{idx}"
            
    df_t.rename(index=nuevos_indices, inplace=True)
    dfs_batchs.append(df_t)

if dfs_batchs:
    print("\n[INFO] Uniendo matrices de los tres batchs...")
    df_final = pd.concat(dfs_batchs, axis=0, join='outer')
    df_final.fillna(0, inplace=True)
    
    meta_df = pd.read_csv(CLASS_DIR.parent / "metadata.csv")
    samples_validas = meta_df['sample_id'].dropna().tolist()
    df_final = df_final[df_final.index.isin(samples_validas)]
    
    out_file = OUT_DIR / "otu_table.csv"
    df_final.index.name = "sample_id"
    df_final.to_csv(out_file)
    
    print("==========================================================")
    print(f"¡Éxito! Tabla unificada guardada en: {out_file}")
    print(f"Dimensiones finales: {df_final.shape[0]} samples, {df_final.shape[1]} especies taxonómicas.")
    print("El proyecto ahora está completamente listo para el script de Deep Learning.")
    print("==========================================================")
else:
    print("No se pudieron combinar las tablas. Revise los mensajes de error.")
