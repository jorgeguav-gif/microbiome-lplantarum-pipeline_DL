"""
@author: Jorge Luis Gutiérrez-Ávila
@institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
@orcid: https://orcid.org/0000-0003-1630-954X
@github: jorgeguav-gif
"""

import pandas as pd
import numpy as np
import os
import multiprocessing
import urllib.request
from micom import Community
from micom.workflows import build

BASE_DIR = "/scratch/users/sgonzalezh852/seq/Seq_Jorge/analysis"
MODEL_DIR = os.path.join(BASE_DIR, "06_metabolic_modeling")
OTU_FILE = os.path.join(BASE_DIR, "03_classification/combined/otu_table.csv")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.csv")

os.makedirs(os.path.join(MODEL_DIR, "models"), exist_ok=True)

def parse_tax(x):
    parts = str(x).split(';')
    full_species = parts[-1].strip() if len(parts) > 0 else 'Unknown'
    genus = parts[-2].strip() if len(parts) > 1 else 'Unknown'
    family = parts[-3].strip() if len(parts) > 2 else 'Unknown'
    order = parts[-4].strip() if len(parts) > 3 else 'Unknown'
    
    return pd.Series({
        'order': order,
        'family': family,
        'genus': genus,
        'species': full_species,
        'id': full_species.replace(" ", "_")
    })

def simulate_sample(args):
    sample_id, taxa_df, db_file = args
    print(f"Simulando sample: {sample_id}")
    try:
        com = Community(taxa_df, model_db=db_file, id=sample_id, solver='glpk')
        
        for ex in com.exchanges:
            ex.lower_bound = -1000.0
            
        sol = com.optimize(fluxes=True, pfba=False)
        
        if sol is None or sol.fluxes is None:
            print(f"La optimizacion FBA no encontro solucion viable (o devolvio None) para la sample {sample_id}.")
            return None, None
            
        rates = sol.members.copy()
        rates['sample_id'] = sample_id
        
        ex_fluxes = sol.fluxes[sol.fluxes.index.str.startswith('EX_') & sol.fluxes.index.str.endswith('_m')].copy()
        ex_fluxes['sample_id'] = sample_id
        
        return rates, ex_fluxes
        
    except Exception as e:
        print(f"Error processing {sample_id}: {e}")
        return None, None

def main():
    db_file = os.path.join(MODEL_DIR, "agora201_refseq216_species_1.qza")
    
    if not os.path.exists(db_file):
        print("Descargando base de datos AGORA2 (RefSeq216, Species) desde Zenodo...")
        urllib.request.urlretrieve("https://zenodo.org/records/7739096/files/agora201_refseq216_species_1.qza", db_file)
    
    print("Loading table de abundancias OTU...")
    otu_table = pd.read_csv(OTU_FILE, index_col=0)
    
    sample_cols = [c for c in otu_table.columns if not c in ['taxonomy', 'tax', 'id', 'species']]
    
    all_rates = []
    all_exchanges = []
    
    args_list = []
    for sample_id in sample_cols:
        sample_abund = otu_table[sample_id]
        sample_abund = sample_abund[sample_abund > 0]
        
        if sample_abund.sum() == 0:
            continue
            
        sample_abund = sample_abund / sample_abund.sum()
        sample_abund = sample_abund[sample_abund > 0.001]
        
        if sample_abund.empty:
            continue
            
        taxa_df = sample_abund.index.to_series().apply(parse_tax)
        taxa_df['abundance'] = sample_abund.values
        taxa_df['sample_id'] = sample_id
        
        args_list.append((sample_id, taxa_df, db_file))
        
    print(f"Construyendo modelos comunitarios con MICOM (AGORA2) para {len(args_list)} muestras...")
    
    threads = min(32, multiprocessing.cpu_count())
    with multiprocessing.Pool(threads) as pool:
        results = pool.map(simulate_sample, args_list)
        
    for rates, ex_fluxes in results:
        if rates is not None:
            all_rates.append(rates)
        if ex_fluxes is not None:
            all_exchanges.append(ex_fluxes)
            
    if all_rates and all_exchanges:
        print("Saving results de flujo y crecimiento...")
        final_rates = pd.concat(all_rates)
        final_rates.to_csv(os.path.join(MODEL_DIR, 'growth_rates.csv'))
        
        final_exchanges = pd.concat(all_exchanges)
        final_exchanges.to_csv(os.path.join(MODEL_DIR, 'exchanges.csv'))
        print("Phase 3 finalizada exitosamente.")
    else:
        print("Error crítico: Ninguna sample pudo resolverse matemáticamente.")

if __name__ == '__main__':
    main()
