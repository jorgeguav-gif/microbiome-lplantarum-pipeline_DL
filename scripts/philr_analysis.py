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
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from Bio import Phylo

mpl.rcParams['svg.fonttype'] = 'none'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--otu_table", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--figures_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print("Cargando datos...")
    otu = pd.read_csv(args.otu_table, index_col=0)
    meta = pd.read_csv(args.metadata, index_col='sample_id')
    
    taxa = [
        "Bacteria;Bacillati;Actinomycetota;Actinomycetes;Mycobacteriales;Corynebacteriaceae;Corynebacterium;Corynebacterium dentalis",
        "Bacteria;Bacillati;Actinomycetota;Coriobacteriia;Coriobacteriales;Coriobacteriaceae;Parvibacter;Parvibacter caecicola",
        "Bacteria;Bacillati;Actinomycetota;Coriobacteriia;Eggerthellales;Eggerthellaceae;Adlercreutzia;Adlercreutzia caecimuris",
        "Bacteria;Bacillati;Actinomycetota;Coriobacteriia;Eggerthellales;Eggerthellaceae;Adlercreutzia;Adlercreutzia mucosicola",
        "Bacteria;Bacillati;Actinomycetota;Coriobacteriia;Eggerthellales;Eggerthellaceae;Adlercreutzia;Adlercreutzia muris",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Enterococcaceae;Enterococcus;Enterococcus avium",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Enterococcaceae;Enterococcus;Enterococcus faecium",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Enterococcaceae;Enterococcus;Enterococcus hirae",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactiplantibacillus;Lactiplantibacillus plantarum",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus acidophilus",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus amylovorus",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus crispatus",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus gallinarum",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus gasseri",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus hominis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus intestinalis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus johnsonii",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus kitasatonis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus paragasseri",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus psittaci",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus rodentium",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus taiwanensis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Ligilactobacillus;Ligilactobacillus animalis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Ligilactobacillus;Ligilactobacillus murinus",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Limosilactobacillus;Limosilactobacillus agrestis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Limosilactobacillus;Limosilactobacillus albertensis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Limosilactobacillus;Limosilactobacillus antri",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Limosilactobacillus;Limosilactobacillus balticus",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Limosilactobacillus;Limosilactobacillus caviae",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Limosilactobacillus;Limosilactobacillus coleohominis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Limosilactobacillus;Limosilactobacillus portuensis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Limosilactobacillus;Limosilactobacillus reuteri",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Limosilactobacillus;Limosilactobacillus rudii",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Limosilactobacillus;Limosilactobacillus urinaemulieris",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Limosilactobacillus;[Lactobacillus] timonensis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Streptococcaceae;Lactococcus;Lactococcus formosensis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Streptococcaceae;Lactococcus;Lactococcus garvieae",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Streptococcaceae;Lactococcus;Lactococcus taiwanensis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Streptococcaceae;Streptococcus;Streptococcus acidominimus",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Streptococcaceae;Streptococcus;Streptococcus danieliae",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Streptococcaceae;Streptococcus;Streptococcus varani",
        "Bacteria;Bacillati;Bacillota;Clostridia;Christensenellales;Christensenellaceae;Guopingia;Guopingia tenuis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Acutalibacteraceae;Acutalibacter;Acutalibacter muris",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Acutalibacteraceae;Caproiciproducens;Caproiciproducens galactitolivorans",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Butyricicoccaceae;Agathobaculum;Agathobaculum butyriciproducens",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Butyricicoccaceae;Butyricicoccus;Butyricicoccus pullicaecorum",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Clostridiaceae;Clostridium;Clostridium celatum",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Clostridiaceae;Clostridium;Clostridium disporicum",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Clostridiaceae;Clostridium;Clostridium massiliamazoniense",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Clostridiaceae;Clostridium;Clostridium phoceensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Clostridiaceae;Clostridium;Clostridium porci",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Clostridiaceae;Clostridium;Clostridium saudiense",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Clostridiaceae;Mordavella;Mordavella massiliensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriaceae;Anaerofustis;Anaerofustis stercorihominis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriaceae;Eubacterium;Eubacterium coprostanoligenes",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriaceae;Eubacterium;Eubacterium maltosivorans",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriales Family XIII. Incertae Sedis;Ihubacter;Ihubacter massiliensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriales Family XIII. Incertae Sedis;Lentihominibacter;Lentihominibacter hominis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriales Family XIII. Incertae Sedis;Zhenpiania;Zhenpiania hominis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriales_Incertae_sedis;Colidextribacter;Colidextribacter massiliensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriales_Incertae_sedis;Evtepia;Evtepia gabavorous",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriales_Incertae_sedis;Intestinimonas;Intestinimonas butyriciproducens",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriales_Incertae_sedis;Intestinimonas;Intestinimonas gabonensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriales_Incertae_sedis;Intestinimonas;Intestinimonas timonensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriales_Incertae_sedis;Massilistercora;Massilistercora timonensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Anaerotruncus;Anaerotruncus colihominis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Flavonifractor;Flavonifractor plautii",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Lawsonibacter;Lawsonibacter asaccharolyticus",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Marasmitruncus;Marasmitruncus massiliensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Neglectibacter;Neglectibacter timonensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Oscillibacter;Oscillibacter massiliensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Oscillibacter;Oscillibacter valericigenes",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Pseudoflavonifractor;Pseudoflavonifractor capillosus",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Pseudoflavonifractor;Pseudoflavonifractor gallinarum",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Pseudoflavonifractor;Pseudoflavonifractor phocaeensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Pusillibacter;Pusillibacter faecalis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Ruminococcus;Ruminococcus champanellensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Ruminococcus;Ruminococcus flavefaciens",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Ruthenibacterium;Ruthenibacterium lactatiformans",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Acetatifactor;Acetatifactor muris",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Anaerostipes;Anaerostipes hominis (ex Lee et al. 2021)",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia caecimuris",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia coccoides",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia glucerasea",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia hominis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia producta",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia pseudococcoides",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia schinkii",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Enterocloster;Enterocloster clostridioformis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Faecalicatena;Faecalicatena absiana",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Faecalicatena;Faecalicatena contorta",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Faecalicatena;Faecalicatena faecalis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Faecalicatena;Faecalicatena orotica",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Faecalimonas;Faecalimonas umbilicata",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Fusimonas;Fusimonas intestini",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Hominisplanchenecus;Hominisplanchenecus faecis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Jutongia;Jutongia hominis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Jutongia;Jutongia huaianensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Kineothrix;Kineothrix alysoides",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Lachnoclostridium;[Clostridium] scindens",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Lacrimispora;Lacrimispora aerotolerans",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Lacrimispora;Lacrimispora celerecrescens",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Lacrimispora;Lacrimispora saccharolytica",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Lacrimispora;Lacrimispora xylanolytica",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Mediterraneibacter;Mediterraneibacter butyricigenes",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Mediterraneibacter;Mediterraneibacter faecis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Mediterraneibacter;Mediterraneibacter glycyrrhizinilyticus",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Mediterraneibacter;Mediterraneibacter gnavus",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Mediterraneibacter;[Ruminococcus] torques",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Qiania;Qiania dongpingensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Roseburia;Roseburia hominis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Roseburia;Roseburia intestinalis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Schaedlerella;Schaedlerella arabinosiphila",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Sporofaciens;Sporofaciens musculi",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Variimorphobacter;Variimorphobacter saccharofermentans",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Wansuia;Wansuia hejianensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Peptostreptococcales;Peptostreptococcaceae;Romboutsia;Romboutsia ilealis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Peptostreptococcales;Peptostreptococcaceae;Romboutsia;Romboutsia maritimum",
        "Bacteria;Bacillati;Bacillota;Clostridia;Peptostreptococcales;Peptostreptococcaceae;Romboutsia;Romboutsia timonensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Peptostreptococcales;Peptostreptococcaceae;Terrisporobacter;Terrisporobacter mayombei",
        "Bacteria;Bacillati;Bacillota;Erysipelotrichia;Erysipelotrichales;Coprobacillaceae;Coprobacillus;Coprobacillus cateniformis",
        "Bacteria;Bacillati;Bacillota;Erysipelotrichia;Erysipelotrichales;Coprobacillaceae;Longibaculum;Longibaculum muris",
        "Bacteria;Bacillati;Bacillota;Erysipelotrichia;Erysipelotrichales;Coprobacillaceae;Thomasclavelia;Thomasclavelia cocleata",
        "Bacteria;Bacillati;Bacillota;Erysipelotrichia;Erysipelotrichales;Coprobacillaceae;Thomasclavelia;Thomasclavelia ramosa",
        "Bacteria;Bacillati;Bacillota;Erysipelotrichia;Erysipelotrichales;Erysipelotrichaceae;Faecalibaculum;Faecalibaculum rodentium",
        "Bacteria;Bacillati;Bacillota;Erysipelotrichia;Erysipelotrichales;Erysipelotrichaceae;Holdemania;Holdemania massiliensis",
        "Bacteria;Bacillati;Bacillota;Erysipelotrichia;Erysipelotrichales;Turicibacteraceae;Turicibacter;Turicibacter bilis",
        "Bacteria;Bacillati;Bacillota;Erysipelotrichia;Erysipelotrichales;Turicibacteraceae;Turicibacter;Turicibacter sanguinis",
        "Bacteria;Bacillati;Mycoplasmatota;Mycoplasmatota_Incertae_sedis;Mycoplasmoidales;Metamycoplasmataceae;Mesomycoplasma;Mesomycoplasma moatsii",
        "Bacteria;Bacillati;Mycoplasmatota;Mycoplasmatota_Incertae_sedis;Mycoplasmoidales;Metamycoplasmataceae;Metamycoplasma;Metamycoplasma sualvi",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Bacteroides;Bacteroides acidifaciens",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Bacteroides;Bacteroides caecimuris",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Bacteroides;Bacteroides faecichinchillae",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Bacteroides;Bacteroides intestinalis",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Bacteroides;Bacteroides rodentium",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Bacteroides;Bacteroides uniformis",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Phocaeicola;Phocaeicola faecalis",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Phocaeicola;Phocaeicola massiliensis",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Phocaeicola;Phocaeicola sartorii",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Phocaeicola;Phocaeicola vulgatus",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Muribaculaceae;Duncaniella;Duncaniella freteri",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Muribaculaceae;Duncaniella;Duncaniella muricolitica",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Muribaculaceae;Duncaniella;Duncaniella muris",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Muribaculaceae;Heminiphilus;Heminiphilus faecis",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Muribaculaceae;Muribaculum;Muribaculum gordoncarteri",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Muribaculaceae;Muribaculum;Muribaculum intestinale",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Muribaculaceae;Paramuribaculum;Paramuribaculum intestinale",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Muribaculaceae;Sangeribacter;Sangeribacter muris",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Odoribacteraceae;Butyricimonas;Butyricimonas virosa",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Rikenellaceae;Alistipes;Alistipes communis",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Rikenellaceae;Alistipes;Alistipes dispar",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Rikenellaceae;Alistipes;Alistipes finegoldii",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Rikenellaceae;Alistipes;Alistipes montrealensis",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Rikenellaceae;Alistipes;Alistipes okayasuensis",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Rikenellaceae;Alistipes;Alistipes onderdonkii",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Rikenellaceae;Alistipes;Alistipes putredinis",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Rikenellaceae;Alistipes;Alistipes senegalensis",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Rikenellaceae;Alistipes;Alistipes shahii",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Rikenellaceae;Alistipes;Alistipes timonensis",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Tannerellaceae;Parabacteroides;Parabacteroides distasonis",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Tannerellaceae;Parabacteroides;Parabacteroides merdae",
        "Bacteria;Pseudomonadati;Campylobacterota;Epsilonproteobacteria;Campylobacterales;Helicobacteraceae;Helicobacter;Helicobacter canadensis",
        "Bacteria;Pseudomonadati;Campylobacterota;Epsilonproteobacteria;Campylobacterales;Helicobacteraceae;Helicobacter;Helicobacter cinaedi",
        "Bacteria;Pseudomonadati;Campylobacterota;Epsilonproteobacteria;Campylobacterales;Helicobacteraceae;Helicobacter;Helicobacter ganmani",
        "Bacteria;Pseudomonadati;Campylobacterota;Epsilonproteobacteria;Campylobacterales;Helicobacteraceae;Helicobacter;Helicobacter mesocricetorum",
        "Bacteria;Pseudomonadati;Campylobacterota;Epsilonproteobacteria;Campylobacterales;Helicobacteraceae;Helicobacter;Helicobacter muridarum",
        "Bacteria;Pseudomonadati;Campylobacterota;Epsilonproteobacteria;Campylobacterales;Helicobacteraceae;Helicobacter;Helicobacter typhlonius",
        "Bacteria;Pseudomonadati;Deferribacterota;Deferribacteres;Deferribacterales;Mucispirillaceae;Mucispirillum;Mucispirillum schaedleri",
        "Bacteria;Pseudomonadati;Pseudomonadota;Betaproteobacteria;Burkholderiales;Sutterellaceae;Parasutterella;Parasutterella excrementihominis",
        "Bacteria;Pseudomonadati;Pseudomonadota;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Escherichia;Escherichia fergusonii",
        "Bacteria;Pseudomonadati;Pseudomonadota;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Escherichia;Escherichia marmotae",
        "Bacteria;Pseudomonadati;Pseudomonadota;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Shigella;Shigella flexneri",
        "Bacteria;Pseudomonadati;Pseudomonadota;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Shigella;Shigella sonnei",
        "Bacteria;Pseudomonadati;Thermodesulfobacteriota;Desulfovibrionia;Desulfovibrionales;Desulfovibrionaceae;Desulfovibrio;Desulfovibrio porci",
        "Bacteria;Pseudomonadati;Thermodesulfobacteriota;Desulfovibrionia;Desulfovibrionales;Desulfovibrionaceae;Lawsonia;Lawsonia intracellularis",
        "Bacteria;Pseudomonadati;Verrucomicrobiota;Verrucomicrobiia;Verrucomicrobiales;Akkermansiaceae;Akkermansia;Akkermansia muciniphila",
        "Unclassified;Unknown;Unknown;Unknown;Unknown;Unknown;Unknown;Unknown",
        "Bacteria;Bacillati;Actinomycetota;Coriobacteriia;Eggerthellales;Eggerthellaceae;Adlercreutzia;Adlercreutzia hattorii",
        "Bacteria;Bacillati;Bacillota;Bacilli;Bacillales;Bacillaceae;Niallia;Niallia circulans",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Enterococcaceae;Enterococcus;Enterococcus durans",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Enterococcaceae;Enterococcus;Enterococcus gilvus",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Enterococcaceae;Enterococcus;Enterococcus mundtii",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus xujianguonis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Ligilactobacillus;Ligilactobacillus apodemi",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Limosilactobacillus;Limosilactobacillus fastidiosus",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Streptococcaceae;Streptococcus;Streptococcus alactolyticus",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Streptococcaceae;Streptococcus;Streptococcus thermophilus",
        "Bacteria;Bacillati;Bacillota;Bacillota_Incertae_sedis;Bacillota_Incertae_sedis;Bacillota_Incertae_sedis;Negativibacillus;Negativibacillus massiliensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Clostridiaceae;Clostridium;Clostridium jeddahitimonense",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Clostridiaceae;Clostridium;Clostridium perfringens",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Clostridiaceae;Clostridium;Clostridium transplantifaecale",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Clostridiaceae;Lactonifactor;Lactonifactor longoviformis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriaceae;Alkalibaculum;Alkalibaculum bacchi",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriales_Incertae_sedis;Flintibacter;Flintibacter butyricus",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Anaerotruncus;Anaerotruncus massiliensis (ex Togo et al. 2019)",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Fumia;Fumia xinanensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Oscillospiraceae_Incertae_sedis;[Clostridium] methylpentosum",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Oscillospiraceae_Incertae_sedis;[Clostridium] viride",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Ruminococcus;Ruminococcus gauvreauii",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Vescimonas;Vescimonas fastidiosa",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Anaerotignaceae;Anaerotignum;Anaerotignum lactatifermentans",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Anaerostipes;Anaerostipes faecis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia argi",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia brookingsii",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia faecicola",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia faecis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia hansenii",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia luti",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia marasmi",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia wexlerae",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Dorea;Dorea phocaeensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Enterocloster;Enterocloster aldenensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Enterocloster;Enterocloster bolteae",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Faecalicatena;Faecalicatena fissicatena",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Frisingicoccus;Frisingicoccus caecimuris",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Gluceribacter;Gluceribacter canis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Lacrimispora;Lacrimispora algidixylanolytica",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Lacrimispora;Lacrimispora amygdalina",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Merdimonas;Merdimonas faecis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Roseburia;Roseburia faecis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Peptostreptococcales;Peptostreptococcaceae;Paraclostridium;Paraclostridium sordellii",
        "Bacteria;Bacillati;Bacillota;Clostridia;Peptostreptococcales;Peptostreptococcaceae;Romboutsia;Romboutsia hominis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Peptostreptococcales;Peptostreptococcaceae;Romboutsia;Romboutsia lituseburensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Peptostreptococcales;Peptostreptococcaceae;Terrisporobacter;Terrisporobacter petrolearius",
        "Bacteria;Bacillati;Bacillota;Erysipelotrichia;Erysipelotrichales;Erysipelotrichaceae;Erysipelotrichaceae_Incertae_sedis;[Clostridium] innocuum",
        "Bacteria;Bacillati;Bacillota;Negativicutes;Veillonellales;Veillonellaceae;Veillonella;Veillonella seminalis",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Bacteroides;Bacteroides clarus",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Bacteroides;Bacteroides oleiciplenus",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Phocaeicola;Phocaeicola coprocola",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Phocaeicola;Phocaeicola plebeius",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Prevotellaceae;Segatella;Segatella copri",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Rikenellaceae;Alistipes;Alistipes ihumii",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Rikenellaceae;Alistipes;Alistipes indistinctus",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Rikenellaceae;Tidjanibacter;Tidjanibacter massiliensis",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Tannerellaceae;Parabacteroides;Parabacteroides goldsteinii",
        "Bacteria;Pseudomonadati;Pseudomonadota;Alphaproteobacteria;Hyphomicrobiales;Methylobacteriaceae;Microvirga;Microvirga pudoricolor",
        "Bacteria;Pseudomonadati;Pseudomonadota;Betaproteobacteria;Burkholderiales;Sutterellaceae;Parasutterella;Parasutterella secunda",
        "Bacteria;Pseudomonadati;Pseudomonadota;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Escherichia;Escherichia albertii",
        "Bacteria;Pseudomonadati;Pseudomonadota;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Escherichia;Escherichia coli",
        "Bacteria;Pseudomonadati;Pseudomonadota;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Shigella;Shigella boydii",
        "Bacteria;Pseudomonadati;Pseudomonadota;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Shigella;Shigella dysenteriae",
        "Bacteria;Pseudomonadati;Pseudomonadota;Gammaproteobacteria;Pasteurellales;Pasteurellaceae;Muribacter;Muribacter muris",
        "Bacteria;Pseudomonadati;Pseudomonadota;Gammaproteobacteria;Pasteurellales;Pasteurellaceae;Rodentibacter;Rodentibacter pneumotropicus",
        "Bacteria;Bacillati;Actinomycetota;Actinomycetes;Micrococcales;Dermabacteraceae;Brachybacterium;Brachybacterium conglomeratum",
        "Bacteria;Bacillati;Bacillota;Bacilli;Bacillales;Bacillaceae;Alkalicoccobacillus;Alkalicoccobacillus plakortidis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Bacillales;Staphylococcaceae;Staphylococcus;Staphylococcus ureilyticus",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Enterococcaceae;Enterococcus;Enterococcus lactis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Enterococcaceae;Enterococcus;Enterococcus malodoratus",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Enterococcaceae;Enterococcus;Enterococcus villorum",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;Lactobacillus ultunensis",
        "Bacteria;Bacillati;Bacillota;Bacilli;Lactobacillales;Lactobacillaceae;Limosilactobacillus;Limosilactobacillus frumenti",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Clostridiaceae;Clostridium;Clostridium fessum",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriaceae;Eubacterium;Eubacterium ventriosum",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Eubacteriales_Incertae_sedis;Intestinimonas;Intestinimonas massiliensis (ex Afouda et al. 2020)",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Acetivibrio;Acetivibrio ethanolgignens",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Faecalibacterium;Faecalibacterium duncaniae",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Faecalibacterium;Faecalibacterium hattorii",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Harryflintia;Harryflintia acetispora",
        "Bacteria;Bacillati;Bacillota;Clostridia;Eubacteriales;Oscillospiraceae;Marseillibacter;Marseillibacter massiliensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Anaerostipes;Anaerostipes caccae",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Anaerostipes;Anaerostipes hadrus",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Blautia;Blautia stercoris",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Coprococcus;Coprococcus comes",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Cuneatibacter;Cuneatibacter caecimuris",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Enterocloster;Enterocloster hominis (ex Hitch et al. 2024)",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Jingyaoa;Jingyaoa shaoxingensis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Waltera;Waltera intestinalis",
        "Bacteria;Bacillati;Bacillota;Clostridia;Peptostreptococcales;Peptostreptococcaceae;Faecalimicrobium;Faecalimicrobium dakarense",
        "Bacteria;Bacillati;Bacillota;Clostridia;Peptostreptococcales;Peptostreptococcaceae;Paraclostridium;Paraclostridium tenue",
        "Bacteria;Bacillati;Bacillota;Tissierellia;Tissierellales;Tissierellaceae;Tissierella;Tissierella pigra",
        "Bacteria;Bacillati;Mycoplasmatota;Mycoplasmatota_Incertae_sedis;Mycoplasmoidales;Mycoplasmoidaceae;Malacoplasma;Malacoplasma muris",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Bacteroides;Bacteroides stercorirosoris",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Phocaeicola;Phocaeicola faecicola",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Muribaculaceae;Duncaniella;Duncaniella dubosii",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Odoribacteraceae;Butyricimonas;Butyricimonas paravirosa",
        "Bacteria;Pseudomonadati;Bacteroidota;Bacteroidia;Bacteroidales;Rikenellaceae;Millionella;Millionella massiliensis",
        "Bacteria;Pseudomonadati;Campylobacterota;Epsilonproteobacteria;Campylobacterales;Helicobacteraceae;Helicobacter;Helicobacter canicola",
        "Bacteria;Pseudomonadati;Campylobacterota;Epsilonproteobacteria;Campylobacterales;Helicobacteraceae;Helicobacter;Helicobacter hepaticus",
        "Bacteria;Pseudomonadati;Campylobacterota;Epsilonproteobacteria;Campylobacterales;Helicobacteraceae;Helicobacter;Helicobacter monodelphidis",
        "Bacteria;Pseudomonadati;Pseudomonadota;Alphaproteobacteria;Hyphomicrobiales;Aestuariivirgaceae;Aestuariivirga;Aestuariivirga litoralis",
        "Bacteria;Pseudomonadati;Pseudomonadota;Alphaproteobacteria;Sphingomonadales;Sphingomonadaceae;Sphingorhabdus;Sphingorhabdus wooponensis",
        "Bacteria;Pseudomonadati;Pseudomonadota;Betaproteobacteria;Burkholderiales;Sutterellaceae;Sutterella;Sutterella massiliensis",
        "Bacteria;Pseudomonadati;Spirochaetota;Spirochaetia;Brachyspirales;Brachyspiraceae;Brachyspira;Brachyspira hyodysenteriae"
    ]
    
    print("Mapeando árbol filogenético...")
    tree_str = "(Seq_247:0.02864,((((Seq_109:0.06003,Seq_259:0.06767)0.913:0.01529,(Seq_114:0.09357,(Seq_197:0.12439,((Seq_041:0.11648,((((Seq_051:0.00793,(Seq_046:0.00916,(Seq_047:0.00367,Seq_186:0.01167)0.722:0.00090)0.962:0.00907)1.000:0.04659,(Seq_048:0.02044,Seq_187:0.22141)0.704:0.01841)1.000:0.10806,(Seq_053:0.10674,(Seq_055:0.08256,Seq_190:0.11865)0.723:0.03521)1.000:0.12046)0.530:0.01607,((Seq_056:0.04230,(Seq_057:0.05504,Seq_058:0.05389)0.846:0.02953)1.000:0.13864,(Seq_265:0.11682,((Seq_216:0.02962,(Seq_217:0.01398,((Seq_119:0.00375,Seq_219:0.00866)1.000:0.02782,(((Seq_218:0.00906,Seq_263:0.04168)0.929:0.00858,Seq_117:0.02179)0.738:0.00297,(Seq_116:0.01860,Seq_118:0.00364)0.990:0.01180)0.808:0.00990)0.482:0.01507)1.000:0.02881)0.659:0.01209,Seq_264:0.04708)1.000:0.12220)0.999:0.05916)0.982:0.04347)0.026:0.01832)0.998:0.04748,(((((((Seq_161:0.03803,((Seq_274:0.01666,((Seq_273:0.00551,(Seq_163:0.01390,Seq_164:0.03546)0.979:0.01822)0.853:0.00862,(Seq_160:0.01142,Seq_272:0.00146)1.000:0.03187)0.983:0.01107)0.945:0.00978,Seq_159:0.00825)1.000:0.02542)0.492:0.01437,Seq_162:0.00960)1.000:0.30622,(((Seq_271:0.10035,(Seq_229:0.04788,((((Seq_147:0.03203,Seq_151:0.04963)0.961:0.01565,(Seq_148:0.01314,((Seq_155:0.00385,(Seq_149:0.01290,Seq_152:0.02612)0.961:0.00949)0.354:0.00674,(Seq_150:0.00931,(Seq_156:0.01846,Seq_154:0.01032)0.644:0.00548)0.084:0.00482)0.986:0.01600)0.767:0.00784)0.827:0.01555,Seq_153:0.00686)1.000:0.03671,(Seq_227:0.03484,Seq_228:0.02235)1.000:0.03415)0.763:0.02090)1.000:0.05302)0.996:0.05990,(((Seq_230:0.02497,(Seq_157:0.05552,Seq_158:0.02401)0.974:0.02034)1.000:0.06513,((((Seq_139:0.03164,(Seq_138:0.02909,(Seq_137:0.02913,Seq_136:0.04343)0.840:0.01338)0.988:0.02004)0.997:0.03674,(Seq_224:0.02254,(Seq_225:0.03219,Seq_268:0.00977)0.998:0.02661)0.993:0.02778)0.818:0.02458,(((Seq_131:0.03008,((Seq_134:0.00515,Seq_135:0.01813)1.000:0.03433,(Seq_267:0.00836,(Seq_223:0.00863,Seq_133:0.02571)0.922:0.00944)1.000:0.02588)0.918:0.01338)0.941:0.01746,Seq_132:0.02297)0.787:0.01668,Seq_130:0.16277)0.929:0.02231)0.908:0.03014,(Seq_222:0.02525,Seq_226:0.17182)0.356:0.01428)1.000:0.06620)0.845:0.02003,((Seq_146:0.01305,Seq_270:0.02183)1.000:0.15458,((((Seq_140:0.07894,(Seq_141:0.03528,Seq_269:0.02122)0.293:0.01691)1.000:0.04260,Seq_143:0.03423)0.888:0.01671,(Seq_144:0.04734,Seq_145:0.10574)0.945:0.02289)0.577:0.00353,Seq_142:0.04862)1.000:0.08769)0.871:0.03162)0.986:0.04287)1.000:0.28810,(Seq_276:0.09784,(Seq_231:0.05778,Seq_275:0.10353)0.663:0.03364)1.000:0.14175)0.265:0.02012)0.257:0.00846,((Seq_165:0.22952,(Seq_171:0.06211,Seq_172:0.09739)1.000:0.10668)0.641:0.04225,Seq_173:0.32340)0.749:0.01112)0.995:0.04900,(((Seq_174:0.08199,(Seq_277:0.03158,(Seq_232:0.02542,Seq_166:0.11304)0.907:0.02905)0.904:0.02562)1.000:0.10517,((Seq_237:0.02889,Seq_238:0.03319)1.000:0.07947,(Seq_236:0.12127,((Seq_234:0.07086,Seq_233:0.16881)0.875:0.01391,((((Seq_170:0.01139,Seq_235:0.00500)0.802:0.00204,Seq_169:0.00150)0.876:0.00146,Seq_167:0.00055)0.875:0.00200,Seq_168:0.00655)0.788:0.00199)0.961:0.01714)1.000:0.07219)1.000:0.11119)1.000:0.10111,(Seq_278:0.38259,(Seq_239:0.14195,((Seq_002:0.01224,Seq_003:0.02020)0.773:0.00742,(Seq_004:0.01611,(Seq_001:0.03480,Seq_175:0.01351)0.453:0.01613)0.917:0.01398)1.000:0.20730)0.998:0.08513)0.914:0.03251)0.196:0.02395)0.993:0.05619,((Seq_126:0.07793,(((Seq_125:0.10266,(Seq_124:0.17485,Seq_220:0.09012)0.991:0.04183)0.996:0.04357,((Seq_127:0.09146,Seq_196:0.08204)1.000:0.09234,(Seq_266:0.27772,(Seq_128:0.00855,Seq_129:0.02441)1.000:0.16158)0.987:0.05625)0.998:0.05959)0.924:0.02790,((Seq_122:0.02727,Seq_123:0.02343)0.994:0.04172,(Seq_120:0.03957,Seq_121:0.03406)0.994:0.03823)1.000:0.11111)0.975:0.03535)0.795:0.03254,((((Seq_038:0.04014,Seq_040:0.03643)0.126:0.01690,(Seq_184:0.04079,(Seq_039:0.03065,Seq_183:0.01373)0.938:0.01469)0.500:0.00890)0.988:0.03245,(Seq_037:0.02742,(Seq_035:0.01259,Seq_036:0.00054)1.000:0.06114)1.000:0.05846)1.000:0.07448,((((Seq_005:0.00369,(Seq_178:0.00053,Seq_243:0.00147)0.870:0.00161)0.914:0.00667,(((Seq_179:0.00824,Seq_244:0.01756)0.728:0.00474,((Seq_006:0.00055,Seq_242:0.00055)0.993:0.00055,Seq_177:0.04505)0.889:0.00429)0.045:0.00095,Seq_007:0.00167)0.855:0.00338)1.000:0.04180,((Seq_181:0.00462,(Seq_022:0.00853,Seq_023:0.00209)0.991:0.01686)1.000:0.04795,(Seq_008:0.05116,(((((Seq_026:0.02562,(Seq_182:0.00408,(Seq_030:0.06371,Seq_033:0.04191)0.999:0.03455)0.075:0.00101)0.901:0.00393,Seq_246:0.01383)0.345:0.00304,(Seq_029:0.03672,Seq_034:0.01843)0.986:0.01540)0.198:0.01048,((Seq_024:0.00055,(Seq_031:0.00308,Seq_027:0.00308)0.998:0.00066)0.987:0.00945,(Seq_025:0.00204,(Seq_028:0.00999,Seq_032:0.00383)0.848:0.00181)0.921:0.00380)0.884:0.00575)0.996:0.03663,(((Seq_020:0.01343,(Seq_014:0.00958,((Seq_016:0.00368,Seq_021:0.00066)0.837:0.00475,(Seq_013:0.00074,Seq_018:0.00075)0.771:0.00081)0.984:0.01563)0.852:0.00707)1.000:0.03911,(Seq_180:0.02109,(Seq_015:0.01002,(((Seq_011:0.07433,Seq_017:0.00178)0.000:0.00054,Seq_010:0.00207)0.768:0.00079,(Seq_009:0.01581,(Seq_012:0.00106,Seq_245:0.07407)0.855:0.00702)0.956:0.00543)0.996:0.01832)0.975:0.01334)0.999:0.03204)0.952:0.02822,Seq_019:0.02242)1.000:0.06013)0.998:0.03995)0.981:0.02414)0.995:0.02392)0.887:0.01792,(Seq_241:0.06339,(Seq_176:0.03432,Seq_240:0.07659)0.940:0.01744)1.000:0.04030)0.113:0.04379)0.957:0.05360)1.000:0.09477)0.859:0.02344,(Seq_221:0.22571,(((Seq_054:0.11524,(Seq_043:0.04143,(Seq_042:0.05568,Seq_069:0.01703)0.991:0.02908)1.000:0.04253)0.565:0.01327,((((Seq_076:0.03470,Seq_077:0.05534)1.000:0.04316,(Seq_253:0.04905,(Seq_185:0.06140,(Seq_065:0.08003,Seq_068:0.01501)1.000:0.10624)0.662:0.01200)0.996:0.02695)0.874:0.01629,(Seq_192:0.10377,(Seq_078:0.03040,(Seq_251:0.01332,Seq_252:0.02252)1.000:0.07974)0.999:0.04902)0.846:0.01682)0.939:0.01602,Seq_193:0.05774)0.851:0.01422)0.967:0.02184,((Seq_044:0.05824,Seq_045:0.10395)1.000:0.05557,((Seq_071:0.02983,(Seq_254:0.03168,(Seq_070:0.01288,Seq_075:0.02926)0.994:0.02315)0.978:0.02050)1.000:0.03125,(((((Seq_061:0.00067,Seq_249:0.00094)1.000:0.03350,Seq_062:0.04866)0.934:0.01112,((((Seq_049:0.01187,Seq_074:0.04200)0.730:0.00361,Seq_067:0.01691)0.965:0.00595,Seq_072:0.01827)0.525:0.00350,(Seq_066:0.01294,Seq_073:0.00561)0.957:0.00711)0.938:0.00792)0.940:0.00906,Seq_063:0.01690)0.969:0.01107,((Seq_059:0.04378,Seq_194:0.04905)0.501:0.01275,(Seq_060:0.04393,Seq_191:0.06774)0.874:0.01660)0.944:0.01070)0.969:0.01969)1.000:0.05336)0.886:0.02643)1.000:0.07804)0.618:0.03335)0.914:0.01945)0.238:0.03442)1.000:0.06291)0.943:0.01583)0.845:0.00744,(Seq_262:0.04643,(Seq_079:0.11841,(Seq_094:0.02463,Seq_098:0.05372)0.995:0.02636)0.841:0.01813)1.000:0.04214)0.952:0.01101,(((Seq_189:0.04660,Seq_195:0.05116)0.994:0.02937,((Seq_110:0.01273,(Seq_215:0.05643,Seq_111:0.01267)0.263:0.00486)1.000:0.04453,(Seq_210:0.09243,(Seq_250:0.04664,(Seq_096:0.03270,Seq_097:0.01746)1.000:0.04900)0.734:0.00795)0.869:0.00813)0.895:0.01230)0.350:0.00315,((Seq_203:0.04190,(Seq_087:0.02053,(Seq_081:0.02868,(Seq_083:0.02656,(Seq_205:0.03407,Seq_201:0.03442)0.960:0.01199)0.937:0.01528)0.598:0.00759)0.594:0.01014)0.987:0.01999,((Seq_198:0.02156,Seq_202:0.00604)1.000:0.02945,(Seq_257:0.05558,((Seq_082:0.0,Seq_085:0.0):0.01742,(Seq_084:0.01619,(Seq_086:0.01442,Seq_204:0.00890)0.930:0.00706)0.950:0.00954)0.911:0.01346)0.890:0.01046)0.943:0.00825)1.000:0.03131)0.826:0.00546)0.387:0.00475,((((Seq_212:0.01799,(Seq_100:0.00055,Seq_103:0.00835)0.910:0.00439)0.970:0.00862,(Seq_213:0.00810,(Seq_102:0.00145,Seq_101:0.09873)0.818:0.00219)0.737:0.00323)1.000:0.03135,(Seq_188:0.04773,(((Seq_088:0.01184,Seq_208:0.01191)0.920:0.01231,((Seq_207:0.02610,Seq_260:0.00518)0.908:0.01302,Seq_050:0.02448)0.763:0.01441)0.987:0.01593,Seq_115:0.05872)0.884:0.00871)0.881:0.00787)0.971:0.01090,((Seq_199:0.00679,Seq_200:0.01116)1.000:0.04957,(((Seq_248:0.05258,(Seq_256:0.02377,(Seq_080:0.01704,Seq_255:0.01671)1.000:0.03484)1.000:0.03171)0.963:0.01486,(((Seq_258:0.03193,(Seq_211:0.03423,(Seq_093:0.02845,(Seq_206:0.03135,Seq_099:0.08024)0.127:0.00594)0.635:0.00560)0.919:0.00664)0.978:0.00948,((Seq_107:0.02488,(Seq_112:0.06701,Seq_113:0.01908)0.986:0.02359)0.977:0.01640,((((Seq_104:0.03426,Seq_106:0.01892)0.927:0.01026,(Seq_091:0.01376,(Seq_092:0.01460,(Seq_090:0.01095,Seq_209:0.00161)0.948:0.00686)0.874:0.00793)1.000:0.02806)0.743:0.00633,((Seq_108:0.03398,Seq_105:0.00980)0.958:0.01245,Seq_000:0.01678)0.920:0.00952)0.681:0.00393,Seq_089:0.04134)0.488:0.00756)0.233:0.00382)0.895:0.00436,(Seq_064:0.03022,(Seq_052:0.02507,Seq_214:0.02210)0.987:0.02095)1.000:0.02571)0.915:0.00583)0.995:0.01708,(Seq_095:0.01578,Seq_261:0.07654)0.990:0.02929)0.455:0.01044)0.490:0.00145)0.220:0.01034);"
    import io
    tree = Phylo.read(io.StringIO(tree_str), "newick")
    seq_to_tax = {f"Seq_{i:03d}": t for i, t in enumerate(taxa)}
    
    for tip in tree.get_terminals():
        if tip.name in seq_to_tax:
            tip.name = seq_to_tax[tip.name]

    common_taxa = list(set(otu.columns).intersection(set([t.name for t in tree.get_terminals()])))
    otu = otu[common_taxa]
    # =========================================================================
    # =========================================================================
    print("Aplicando imputación multiplicativa para manejo de ceros...")
    otu_pseudo = otu.copy().astype(float)
    for idx, row in otu_pseudo.iterrows():
        zeros = (row == 0).sum()
        if zeros > 0:
            tot = row.sum()
            delta = min(0.5, (0.05 * tot) / zeros) if tot > 0 else 0.5
            
            zero_mask = row == 0
            non_zero_mask = row > 0
            otu_pseudo.loc[idx, zero_mask] = delta
            if tot > 0:
                otu_pseudo.loc[idx, non_zero_mask] = row[non_zero_mask] * (1.0 - (zeros * delta) / tot)
    
    otu_rel = otu_pseudo.div(otu_pseudo.sum(axis=1), axis=0)

    print("Calculating Balances Filogenéticos Evolutivos (PhILR)...")
    balances = {}
    node_idx = 0
    for clade in tree.get_nonterminals():
        children = clade.clades
        if len(children) >= 2:
            left_tips = [t.name for t in children[0].get_terminals() if t.name in common_taxa]
            right_tips = [t.name for t in children[1].get_terminals() if t.name in common_taxa]
            
            if len(left_tips) > 0 and len(right_tips) > 0:
                left_name = left_tips[0].split(';')[-1].replace(' ', '_')
                right_name = right_tips[0].split(';')[-1].replace(' ', '_')
                balance_name = f"Balance_{node_idx}_{left_name}_vs_{right_name}"
                
                g_left = np.exp(np.log(otu_rel[left_tips]).mean(axis=1))
                g_right = np.exp(np.log(otu_rel[right_tips]).mean(axis=1))
                
                r = len(left_tips)
                s = len(right_tips)
                scale = np.sqrt((r * s) / (r + s))
                bal_val = scale * np.log(g_left / g_right)
                
                balances[balance_name] = bal_val
                node_idx += 1

    df_bal = pd.DataFrame(balances)
    common_samples = df_bal.index.intersection(meta.index)
    df_bal = df_bal.loc[common_samples]
    meta = meta.loc[common_samples]
    
    print("Generating PCA de balances...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(df_bal)
    var_exp = pca.explained_variance_ratio_ * 100
    df_pca = pd.DataFrame(coords, columns=['PC1', 'PC2'], index=df_bal.index).join(meta)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='tratamiento', style='sexo', s=150,
                    palette={"Control":"#888888", "G7":"#E69F00", "LM20":"#56B4E9", "PS128":"#009E73"}, ax=ax)
    plt.title("PCA de Balances Filogenéticos Evolutivos", fontsize=16)
    plt.xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=12)
    plt.ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    fig.savefig(os.path.join(args.figures_dir, "philr_pca.svg"), format='svg', bbox_inches='tight')
    plt.close(fig)

    print("Identificando top balances usando Regresión Lineal Multivariada ajustada por sexo...")
    resultados_reg = []
    
    try:
        import statsmodels.formula.api as smf
        has_sm = True
    except ImportError:
        has_sm = False
        print("ADVERTENCIA: statsmodels no instalado. Usando Kruskal-Wallis como respaldo.")
    
    for col in df_bal.columns:
        if has_sm:
            df_model = pd.DataFrame({
                'Balance': df_bal[col],
                'Treatment': meta['tratamiento'],
                'Sex': meta['sexo']
            })
            try:
                model = smf.ols("Balance ~ C(Treatment, Treatment('Control')) + C(Sex)", data=df_model).fit()
                pvals_tratamiento = [p for name, p in zip(model.pvalues.index, model.pvalues) if 'Treatment' in name]
                best_pval = min(pvals_tratamiento) if pvals_tratamiento else 1.0
                resultados_reg.append({'Balance': col, 'p_value': best_pval, 'Method': 'OLS_Adjusted'})
            except:
                pass
        else:
            grupos = [df_bal[col][meta['tratamiento'] == t].values for t in meta['tratamiento'].unique()]
            try:
                stat, p_val = kruskal(*grupos)
                resultados_reg.append({'Balance': col, 'H_stat': stat, 'p_value': p_val, 'Method': 'Kruskal-Wallis'})
            except:
                pass

    df_kw = pd.DataFrame(resultados_reg).sort_values('p_value')
    df_kw['FDR'] = df_kw['p_value'] * len(df_kw) / np.arange(1, len(df_kw)+1)
    df_kw.to_csv(os.path.join(args.output_dir, "philr_results.csv"), index=False)
    
    if not df_kw.empty:
        top_balance = df_kw.iloc[0]['Balance']
        print(f"The most differential balance is: {top_balance}")
        
        plot_df = df_bal[[top_balance]].join(meta)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.boxplot(data=plot_df, x='tratamiento', y=top_balance, hue='tratamiento', dodge=False,
                    palette={"Control":"#888888", "G7":"#E69F00", "LM20":"#56B4E9", "PS128":"#009E73"}, ax=ax)
        sns.stripplot(data=plot_df, x='tratamiento', y=top_balance, color=".2", alpha=0.6, size=6, ax=ax)
        
        plt.title("Top Balance Filogenético Diferencial", fontsize=16)
        plt.suptitle(top_balance, fontsize=10, y=0.92, color='gray')
        plt.ylabel("Log-Ratio Isométrico (Balance)", fontsize=12)
        plt.xlabel("Treatment", fontsize=12)
        if ax.get_legend():
            ax.get_legend().remove()
        
        fig.savefig(os.path.join(args.figures_dir, "philr_top_balances.svg"), format='svg', bbox_inches='tight')
        plt.close(fig)

    print("Phase 2 (PhILR in Python) successfully completed.")

if __name__ == "__main__":
    main()
