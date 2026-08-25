"""
# @author: Jorge Luis Gutiérrez-Ávila
# @institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
# @orcid: https://orcid.org/0000-0003-1630-954X
# @github: jorgeguav-gif
"""

import os
import gzip
import random
import argparse
import pandas as pd
from Bio import SeqIO
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML

def main():
    parser = argparse.ArgumentParser(description="Muestrea y realiza BLAST de lecturas Unclassified")
    parser.add_argument("--fastq", required=True, help="Ruta al archivo fastq.gz unclassified")
    parser.add_argument("--output", required=True, help="Ruta del archivo CSV de salida")
    parser.add_argument("--n_seqs", type=int, default=30, help="Número de secuencias a muestrear")
    args = parser.parse_args()

    print(f"[INFO] Extrayendo secuencias de: {args.fastq}")
    
    # Extraer todas las secuencias
    records = []
    with gzip.open(args.fastq, "rt") as handle:
        for record in SeqIO.parse(handle, "fastq"):
            records.append(record)
            
    print(f"[INFO] Se encontraron {len(records)} secuencias en total.")
    
    # Muestrear al azar
    n_samples = min(args.n_seqs, len(records))
    sampled_records = random.sample(records, n_samples)
    print(f"[INFO] Se seleccionaron {n_samples} secuencias al azar para BLAST remoto.")
    
    results = []
    
    # Run BLAST via web API (no need for local DB)
    for i, record in enumerate(sampled_records):
        print(f"[{i+1}/{n_samples}] BLASTing {record.id} (Longitud: {len(record.seq)} pb)...")
        try:
            # qblast(programa, base de datos, secuencia)
            result_handle = NCBIWWW.qblast("blastn", "nt", record.seq, hitlist_size=1)
            blast_record = NCBIXML.read(result_handle)
            
            if len(blast_record.alignments) > 0:
                alignment = blast_record.alignments[0]
                hsp = alignment.hsps[0]
                identidad = (hsp.identities / hsp.align_length) * 100
                hit_title = alignment.title.split("|")[-1].strip()
                
                results.append({
                    "Query_ID": record.id,
                    "Top_Hit": hit_title,
                    "Identidad_pct": round(identidad, 2),
                    "E_value": hsp.expect,
                    "Alineamiento_len": hsp.align_length
                })
                print(f"  -> HIT: {hit_title[:60]}... ({identidad:.1f}%)")
            else:
                results.append({
                    "Query_ID": record.id,
                    "Top_Hit": "NO HITS",
                    "Identidad_pct": 0,
                    "E_value": None,
                    "Alineamiento_len": 0
                })
                print("  -> NO HITS")
        except Exception as e:
            print(f"  -> Error en BLAST: {e}")
            
    # Guardar reporte
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\n[INFO] Análisis completado. Reporte guardado en: {args.output}")

if __name__ == "__main__":
    main()
