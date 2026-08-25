"""
@author: Jorge Luis Gutiérrez-Ávila
@institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
@orcid: https://orcid.org/0000-0003-1630-954X
@github: jorgeguav-gif
"""

import pandas as pd
import numpy as np
import os
from scipy import stats

base_dir = r"./data"
metadata_file = os.path.join(base_dir, 'metadata.csv')
growth_rates_file = os.path.join(base_dir, '06_metabolic_modeling', 'growth_rates.csv')

meta = pd.read_csv(metadata_file, index_col=0)
growth = pd.read_csv(growth_rates_file)

growth = growth[growth['compartments'] != 'm']

growth['treatment'] = growth['sample_id'].apply(lambda x: x.split('_')[0])

species_counts = growth['compartments'].value_counts()
valid_species = species_counts[species_counts > (63 * 0.2)].index
growth = growth[growth['compartments'].isin(valid_species)]

results = []
for especie in valid_species:
    datos_especie = growth[growth['compartments'] == especie]
    
    ctrl_growth = datos_especie[datos_especie['treatment'] == 'CONTROL']['growth_rate'].values
    ps128_growth = datos_especie[datos_especie['treatment'] == 'PS128']['growth_rate'].values
    
    if len(ctrl_growth) >= 3 and len(ps128_growth) >= 3:
        stat, pval = stats.kruskal(ctrl_growth, ps128_growth)
        mean_ctrl = np.mean(ctrl_growth)
        mean_ps128 = np.mean(ps128_growth)
        fc = (mean_ps128 + 1e-9) / (mean_ctrl + 1e-9)
        
        results.append({
            'Especie': especie,
            'Media_Control': mean_ctrl,
            'Media_PS128': mean_ps128,
            'Fold_Change': fc,
            'p_value': pval
        })

res_df = pd.DataFrame(results).dropna(subset=['p_value'])
res_df['p_adj'] = stats.false_discovery_control(res_df['p_value'])
res_df = res_df.sort_values('p_value')

print("--- TOP ESPECIES CON CRECIMIENTO DIFERENCIAL (PS128 vs CONTROL) ---")
print(res_df.head(10).to_string(index=False))

resultados_lm20 = []
for especie in valid_species:
    datos_especie = growth[growth['compartments'] == especie]
    ctrl_growth = datos_especie[datos_especie['treatment'] == 'CONTROL']['growth_rate'].values
    lm20_growth = datos_especie[datos_especie['treatment'] == 'LM20']['growth_rate'].values
    if len(ctrl_growth) >= 3 and len(lm20_growth) >= 3:
        stat, pval = stats.kruskal(ctrl_growth, lm20_growth)
        fc = (np.mean(lm20_growth) + 1e-9) / (np.mean(ctrl_growth) + 1e-9)
        resultados_lm20.append({'Especie': especie, 'Media_Control': np.mean(ctrl_growth), 'Media_LM20': np.mean(lm20_growth), 'Fold_Change': fc, 'p_value': pval})

res_lm20_df = pd.DataFrame(resultados_lm20)
if len(res_lm20_df) > 0:
    res_lm20_df = res_lm20_df.sort_values('p_value')
    print("\n--- TOP ESPECIES CON CRECIMIENTO DIFERENCIAL (LM20 vs CONTROL) ---")
    print(res_lm20_df.head(5).to_string(index=False))

