"""
# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif
"""

import os
import time
import pandas as pd
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Configura tu email para NCBI Entrez
Entrez.email = "tu_email@institucion.edu"

def clean_tax_name(tax_string):
    parts = str(tax_string).split(';')
    if len(parts) > 0:
        return parts[-1].strip()
    return tax_string

def fetch_16s_sequence(species_name):
    """Busca la secuencia 16S ribosomal de una especie en NCBI."""
    search_term = f"{species_name}[Organism] AND 16S ribosomal RNA[Title] AND 1000:2000[Sequence Length]"
    try:
        # Buscar
        handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=1)
        record = Entrez.read(handle)
        handle.close()
        
        if record["IdList"]:
            # Descargar
            seq_id = record["IdList"][0]
            fetch_handle = Entrez.efetch(db="nucleotide", id=seq_id, rettype="fasta", retmode="text")
            seq_record = SeqIO.read(fetch_handle, "fasta")
            fetch_handle.close()
            return str(seq_record.seq)
        else:
            return None
    except Exception as e:
        print(f"  [ERROR] Falla en conexión NCBI para {species_name}: {e}")
        return None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    otu_path = os.path.join(project_dir, "03_classification", "combined", "otu_table.csv")
    out_fasta = os.path.join(project_dir, "04_statistics", "picrust2_input.fasta")
    out_tsv = os.path.join(project_dir, "04_statistics", "picrust2_input.tsv")
    
    print("[INFO] Cargando tabla OTU...")
    otu = pd.read_csv(otu_path, index_col=0).T
    
    # Limpiar nombres (eliminando Unclassified)
    if 'Unclassified' in otu.index:
        otu = otu.drop('Unclassified')
        
    species_list = [clean_tax_name(idx) for idx in otu.index]
    otu.index = species_list
    
    # Agrupar especies duplicadas por si acaso
    otu = otu.groupby(otu.index).sum()
    
    print(f"[INFO] Procesando {len(otu.index)} especies únicas...")
    
    records = []
    valid_species = []
    
    for i, sp in enumerate(otu.index):
        print(f"[{i+1}/{len(otu.index)}] Buscando 16S para: {sp}...")
        seq = fetch_16s_sequence(sp)
        if seq:
            # Crear ID seguro sin espacios
            safe_id = sp.replace(" ", "_").replace("[", "").replace("]", "")
            records.append(SeqRecord(Seq(seq), id=safe_id, description=""))
            valid_species.append(sp)
        else:
            print(f"  -> No se encontró secuencia de referencia. Se omitirá en PICRUSt2.")
        time.sleep(0.5) # Respetar límites de API NCBI
        
    # Guardar FASTA
    SeqIO.write(records, out_fasta, "fasta")
    print(f"\n[INFO] FASTA guardado con {len(records)} secuencias: {out_fasta}")
    
    # Filtrar matriz de abundancia solo para las que encontramos secuencia
    otu_valid = otu.loc[valid_species]
    otu_valid.index = [sp.replace(" ", "_").replace("[", "").replace("]", "") for sp in otu_valid.index]
    
    # Guardar TSV para PICRUSt2 (formato compatible con BIOM)
    otu_valid.index.name = '#OTU ID'
    otu_valid.to_csv(out_tsv, sep='\t')
    print(f"[INFO] Matriz TSV guardada: {out_tsv}")

if __name__ == "__main__":
    main()
