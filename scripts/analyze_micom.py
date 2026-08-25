"""
@author: Jorge Luis Gutiérrez-Ávila
@institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
@orcid: https://orcid.org/0000-0003-1630-954X
@github: jorgeguav-gif
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy import stats

base_dir = r"./data"
metadata_file = os.path.join(base_dir, 'metadata.csv')
exchanges_file = os.path.join(base_dir, '06_metabolic_modeling', 'exchanges.csv')
growth_rates_file = os.path.join(base_dir, '06_metabolic_modeling', 'growth_rates.csv')

try:
    meta = pd.read_csv(metadata_file, index_col=0)
    exch = pd.read_csv(exchanges_file)
    growth = pd.read_csv(growth_rates_file)
    
    print("--- ARCHIVOS CARGADOS CORRECTAMENTE ---")
    print(f"Samples en exchanges: {exch['sample_id'].nunique()}")
    print(f"Samples en growth_rates: {growth['sample_id'].nunique()}")
    
    growth = growth[growth['compartments'] != 'm']

    print("\n--- RESUMEN DE TABLA DE TASAS DE CRECIMIENTO ---")
    print(growth.head())
    
    print("\n--- RESUMEN DE TABLA DE FLUJOS (EXCHANGES) ---")
    print(exch.head())

except Exception as e:
    print(f"Error processing: {e}")

