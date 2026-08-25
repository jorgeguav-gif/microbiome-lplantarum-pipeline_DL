#!/usr/bin/env python3
"""
@author: Jorge Luis Gutiérrez-Ávila
@institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
@orcid: https://orcid.org/0000-0003-1630-954X
@github: jorgeguav-gif
"""

import argparse
import csv
import gzip
import os
import sys
import time
from pathlib import Path
from statistics import mean, median

# =============================================================================
# =============================================================================

def calcular_qscore_medio(quality_string: str) -> float:
    
    if not quality_string:
        return 0.0
    total = sum(ord(c) - 33 for c in quality_string)
    return total / len(quality_string)

def parse_fastq_gzip(filepath: str):
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        while True:
            header = f.readline().rstrip('\n')
            if not header:
                break  # Fin del file
            sequence = f.readline().rstrip('\n')
            separator = f.readline().rstrip('\n')
            quality = f.readline().rstrip('\n')

            if not header.startswith('@'):
                print(f"  [ADVERTENCIA] Encabezado inesperado en {filepath}: {header[:50]}",
                      file=sys.stderr)
                continue

            yield header, sequence, separator, quality

def encontrar_directorio_fastq_pass(input_path: str) -> str:
    
    input_path = os.path.abspath(input_path)

    if os.path.basename(input_path) == 'fastq_pass':
        return input_path

    direct = os.path.join(input_path, 'fastq_pass')
    if os.path.isdir(direct):
        return direct

    for root, dirs, _ in os.walk(input_path):
        depth = root.replace(input_path, '').count(os.sep)
        if depth > 3:
            dirs.clear()
            continue
        if 'fastq_pass' in dirs:
            return os.path.join(root, 'fastq_pass')

    raise FileNotFoundError(
        f"Not found el directorio 'fastq_pass' dentro de:\n  {input_path}\n"
        f"Verifique que la ruta apunte al directorio de la corrida de secuenciación."
    )

def obtener_barcodes(fastq_pass_dir: str) -> list:
    
    barcodes = []
    for entry in sorted(os.listdir(fastq_pass_dir)):
        full_path = os.path.join(fastq_pass_dir, entry)
        if os.path.isdir(full_path) and entry.lower().startswith('barcode'):
            barcodes.append(full_path)
    return barcodes

def listar_fastq_gz(barcode_dir: str) -> list:
    
    files = []
    for entry in sorted(os.listdir(barcode_dir)):
        if entry.endswith('.fastq.gz'):
            files.append(os.path.join(barcode_dir, entry))
    return files

# =============================================================================
# =============================================================================

def procesar_barcode(barcode_dir: str, output_dir: str,
                     min_length: int, max_length: int,
                     min_quality: float) -> dict:
    
    barcode_name = os.path.basename(barcode_dir)
    files_gz = listar_fastq_gz(barcode_dir)

    if not files_gz:
        print(f"  [AVISO] No se encontraron files .fastq.gz en {barcode_name}")
        return {
            'barcode': barcode_name,
            'total_reads': 0, 'passed_reads': 0,
            'failed_length': 0, 'failed_quality': 0,
            'mean_length': 0.0, 'median_length': 0.0,
            'mean_qscore': 0.0, 'median_qscore': 0.0
        }

    total_reads = 0
    passed_reads = 0
    failed_length = 0
    failed_quality = 0

    passed_lengths = []
    passed_qscores = []

    output_file = os.path.join(output_dir, f"{barcode_name}_filtered.fastq.gz")

    with gzip.open(output_file, 'wt', encoding='utf-8') as out_fh:
        for file in files_gz:
            for header, sequence, separator, quality in parse_fastq_gzip(file):
                total_reads += 1
                read_len = len(sequence)
                qscore = calcular_qscore_medio(quality)

                if read_len < min_length or read_len > max_length:
                    failed_length += 1
                    continue

                if qscore < min_quality:
                    failed_quality += 1
                    continue

                passed_reads += 1
                passed_lengths.append(read_len)
                passed_qscores.append(qscore)

                out_fh.write(f"{header}\n{sequence}\n{separator}\n{quality}\n")

    stats = {
        'barcode': barcode_name,
        'total_reads': total_reads,
        'passed_reads': passed_reads,
        'failed_length': failed_length,
        'failed_quality': failed_quality,
        'mean_length': round(mean(passed_lengths), 1) if passed_lengths else 0.0,
        'median_length': round(median(passed_lengths), 1) if passed_lengths else 0.0,
        'mean_qscore': round(mean(passed_qscores), 2) if passed_qscores else 0.0,
        'median_qscore': round(median(passed_qscores), 2) if passed_qscores else 0.0,
    }

    pct = (passed_reads / total_reads * 100) if total_reads > 0 else 0.0

    print(f"  {barcode_name}: {total_reads:,} total → {passed_reads:,} pasaron "
          f"({pct:.1f}%) | descartadas: {failed_length:,} (longitud), "
          f"{failed_quality:,} (calidad)")

    return stats

def escribir_resumen_csv(stats_list: list, output_path: str):
    
    fieldnames = [
        'barcode', 'total_reads', 'passed_reads',
        'failed_length', 'failed_quality',
        'mean_length', 'median_length',
        'mean_qscore', 'median_qscore'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for stats in stats_list:
            writer.writerow(stats)

    print(f"\n✓ Resumen CSV saved at: {output_path}")

# =============================================================================
# =============================================================================

def crear_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(
        description=(
            "Preprocesamiento de lecturas FASTQ de secuenciación 16S rRNA (Oxford Nanopore).\n"
            "Filtra lecturas por longitud y calidad media (Phred Q-score)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplo de uso:\n"
            "  py preprocess_fastq.py --input C:\\ruta\\al\\Batch1 "
            "--output C:\\ruta\\salida --min-length 1000 --max-length 1800 --min-quality 10\n\n"
            "El script buscará automáticamente el directorio fastq_pass/ dentro de --input."
        )
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='Directorio base de la corrida de secuenciación (contiene fastq_pass/).'
    )
    parser.add_argument(
        '--output', '-o', required=True,
        help='Directorio de salida para los files filtrados y el resumen CSV.'
    )
    parser.add_argument(
        '--min-length', type=int, default=1000,
        help='Longitud mínima de lectura en bp (default: 1000).'
    )
    parser.add_argument(
        '--max-length', type=int, default=1800,
        help='Longitud máxima de lectura en bp (default: 1800).'
    )
    parser.add_argument(
        '--min-quality', type=float, default=10.0,
        help='Q-score medio mínimo (Phred) para aceptar una lectura (default: 10).'
    )
    return parser

def main():
    
    parser = crear_parser()
    args = parser.parse_args()

    print("=" * 72)
    print("  PREPROCESAMIENTO DE LECTURAS 16S rRNA — Oxford Nanopore")
    print("=" * 72)
    print(f"  Directorio de entrada : {args.input}")
    print(f"  Directorio de salida  : {args.output}")
    print(f"  Filtro de longitud    : {args.min_length} – {args.max_length} bp")
    print(f"  Filtro de calidad     : Q-score medio ≥ {args.min_quality}")
    print("=" * 72)

    try:
        fastq_pass_dir = encontrar_directorio_fastq_pass(args.input)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n→ Directorio fastq_pass found: {fastq_pass_dir}")

    barcodes = obtener_barcodes(fastq_pass_dir)
    if not barcodes:
        print("[ERROR] No se encontraron directorios de barcode en fastq_pass/.",
              file=sys.stderr)
        sys.exit(1)

    print(f"→ Barcodes founds: {len(barcodes)}")
    for bc in barcodes:
        print(f"    • {os.path.basename(bc)}")

    filtered_dir = os.path.join(args.output, 'filtered_reads')
    os.makedirs(filtered_dir, exist_ok=True)
    print(f"\n→ Lecturas filtradas se guardarán en: {filtered_dir}")

    print("\n" + "-" * 72)
    print("  Processing barcodes...")
    print("-" * 72)

    inicio = time.time()
    all_stats = []

    for i, barcode_dir in enumerate(barcodes, 1):
        barcode_name = os.path.basename(barcode_dir)
        print(f"\n[{i}/{len(barcodes)}] Processing {barcode_name}...")
        stats = procesar_barcode(
            barcode_dir=barcode_dir,
            output_dir=filtered_dir,
            min_length=args.min_length,
            max_length=args.max_length,
            min_quality=args.min_quality
        )
        all_stats.append(stats)

    elapsed = time.time() - inicio

    total_all = sum(s['total_reads'] for s in all_stats)
    passed_all = sum(s['passed_reads'] for s in all_stats)
    failed_len_all = sum(s['failed_length'] for s in all_stats)
    failed_qual_all = sum(s['failed_quality'] for s in all_stats)
    pct_all = (passed_all / total_all * 100) if total_all > 0 else 0.0

    print("\n" + "=" * 72)
    print("  RESUMEN GLOBAL")
    print("=" * 72)
    print(f"  Lecturas totales      : {total_all:,}")
    print(f"  Lecturas aprobadas    : {passed_all:,} ({pct_all:.1f}%)")
    print(f"  Descartadas (longitud): {failed_len_all:,}")
    print(f"  Descartadas (calidad) : {failed_qual_all:,}")
    print(f"  Tiempo de ejecución   : {elapsed:.1f} segundos")
    print("=" * 72)

    csv_path = os.path.join(args.output, 'preprocessing_summary.csv')
    escribir_resumen_csv(all_stats, csv_path)

    print(f"\n✓ Preprocesamiento completed exitosamente.")
    print(f"  Files filtrados en : {filtered_dir}")
    print(f"  Resumen CSV en        : {csv_path}")

if __name__ == '__main__':
    main()
