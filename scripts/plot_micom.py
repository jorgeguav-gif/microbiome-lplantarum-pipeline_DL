"""
@author: Jorge Luis Gutiérrez-Ávila
@institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
@orcid: https://orcid.org/0000-0003-1630-954X
@github: jorgeguav-gif
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

base_dir = r"./data"
metadata_file = os.path.join(base_dir, 'metadata.csv')
growth_rates_file = os.path.join(base_dir, '06_metabolic_modeling', 'growth_rates.csv')
figures_dir = os.path.join(base_dir, 'figures')

os.makedirs(figures_dir, exist_ok=True)

growth = pd.read_csv(growth_rates_file)
growth = growth[growth['compartments'] != 'm']
growth['treatment'] = growth['sample_id'].apply(lambda x: x.split('_')[0])

especies_clave = ['Lactobacillus_crispatus', 'Ligilactobacillus_murinus', 'Bacteroides_acidifaciens', 'Phocaeicola_sartorii']
df_plot = growth[growth['compartments'].isin(especies_clave)]

df_plot['Especie'] = df_plot['compartments'].str.replace('_', ' ')

palette = {'CONTROL': '#3498db', 'LM20': '#2ecc71', 'G7': '#f39c12', 'PS128': '#e74c3c'}
markers = {'CONTROL': 'o', 'LM20': 's', 'G7': '^', 'PS128': 'D'}

plt.style.use('default')
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7))

ax = sns.boxplot(x='Especie', y='growth_rate', hue='treatment', data=df_plot, 
                 palette=palette, fliersize=0, boxprops=dict(alpha=0.5))

for trat in df_plot['treatment'].unique():
    subset = df_plot[df_plot['treatment'] == trat]
    pass

sns.stripplot(x='Especie', y='growth_rate', hue='treatment', data=df_plot, 
              dodge=True, alpha=0.9, palette=palette, size=7, linewidth=1, edgecolor='black', legend=False)

from matplotlib.markers import MarkerStyle
paths = {'CONTROL': MarkerStyle('o'), 'LM20': MarkerStyle('s'), 'G7': MarkerStyle('^'), 'PS128': MarkerStyle('D')}

plt.title('Resiliencia Metabólica Intestínal:\nTasas de Crecimiento Máximas Teóricas (In Silico)', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Tasa de Crecimiento Predicha (mmol/gDW/h)', fontsize=14)
plt.xlabel('')
plt.xticks(fontstyle='italic', fontsize=12)

handles, labels = ax.get_legend_handles_labels()
n_trats = len(df_plot['treatment'].unique())
plt.legend(handles[:n_trats], labels[:n_trats], title='Treatment', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'growth_rates_resilience.svg'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, 'growth_rates_resilience.tiff'), dpi=300, bbox_inches='tight')
print("Graficas guardadas en figures/growth_rates_resilience.svg/tiff")
