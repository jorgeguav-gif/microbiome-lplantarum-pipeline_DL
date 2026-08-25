#!/usr/bin/env python3
"""
@author: Jorge Luis Gutiérrez-Ávila
@institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
@orcid: https://orcid.org/0000-0003-1630-954X
@github: jorgeguav-gif
"""

"""
preprocess_fastq.py — Preprocesamiento de lecturas FASTQ de secuenciación 16S rRNA (Oxford Nanopore)

Descripción:
    Lee archivos .fastq.gz organizados por código de barras (barcode) desde una corrida
    de secuenciación MinION, filtra las lecturas por longitud y calidad, y genera
    archivos FASTQ filtrados junto con un resumen estadístico en CSV.

Uso:
    py preprocess_fastq.py --input <directorio_corrida> --output <directorio_salida> \\
        --min-length 1000 --max-length 1800 --min-quality 10

Autor: Pipeline de análisis 16S — Proyecto probióticos CD1
Fecha: 2026-07-16
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
# Funciones auxiliares
# =============================================================================

def calcular_qscore_medio(quality_string: str) -> float:
    """
    Calcula el Q-score medio (Phred) a partir de la cadena de calidad FASTQ.

    Cada carácter ASCII en la cadena de calidad representa un valor Phred:
        Q = ord(carácter) - 33

    Parámetros:
        quality_string: cadena de calidad de una lectura FASTQ (línea 4).

    Retorna:
        Promedio aritmético de los valores Q de la lectura.
    """
    if not quality_string:
        return 0.0
    total = sum(ord(c) - 33 for c in quality_string)
    return total / len(quality_string)


def parse_fastq_gzip(filepath: str):
    """
    Generador que lee un archivo .fastq.gz y produce registros FASTQ uno a uno.

    Cada registro FASTQ consta de exactamente 4 líneas:
        1. Encabezado (comienza con '@')
        2. Secuencia de nucleótidos
        3. Separador (comienza con '+')
        4. Cadena de calidad (codificación Phred+33)

    Parámetros:
        filepath: ruta al archivo .fastq.gz

    Produce:
        Tuplas de (header, sequence, separator, quality)
    """
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        while True:
            # Leer las 4 líneas del registro FASTQ
            header = f.readline().rstrip('\n')
            if not header:
                break  # Fin del archivo
            sequence = f.readline().rstrip('\n')
            separator = f.readline().rstrip('\n')
            quality = f.readline().rstrip('\n')

            # Validación básica del formato
            if not header.startswith('@'):
                print(f"  [ADVERTENCIA] Encabezado inesperado en {filepath}: {header[:50]}",
                      file=sys.stderr)
                continue

            yield header, sequence, separator, quality


def encontrar_directorio_fastq_pass(input_path: str) -> str:
    """
    Busca el directorio 'fastq_pass' dentro de la estructura de la corrida.

    Estrategia de búsqueda (en orden):
        1. Si input_path ES fastq_pass, usarlo directamente.
        2. Si input_path contiene fastq_pass/ como hijo directo.
        3. Buscar recursivamente hasta 3 niveles de profundidad.

    Parámetros:
        input_path: directorio base proporcionado por el usuario.

    Retorna:
        Ruta absoluta al directorio fastq_pass.

    Lanza:
        FileNotFoundError si no se encuentra fastq_pass.
    """
    input_path = os.path.abspath(input_path)

    # Caso 1: el propio directorio se llama fastq_pass
    if os.path.basename(input_path) == 'fastq_pass':
        return input_path

    # Caso 2: hijo directo
    direct = os.path.join(input_path, 'fastq_pass')
    if os.path.isdir(direct):
        return direct

    # Caso 3: búsqueda recursiva (hasta 3 niveles)
    for root, dirs, _ in os.walk(input_path):
        # Limitar profundidad de búsqueda
        depth = root.replace(input_path, '').count(os.sep)
        if depth > 3:
            # No descender más
            dirs.clear()
            continue
        if 'fastq_pass' in dirs:
            return os.path.join(root, 'fastq_pass')

    raise FileNotFoundError(
        f"No se encontró el directorio 'fastq_pass' dentro de:\n  {input_path}\n"
        f"Verifique que la ruta apunte al directorio de la corrida de secuenciación."
    )


def obtener_barcodes(fastq_pass_dir: str) -> list:
    """
    Lista los subdirectorios de barcodes dentro de fastq_pass/.

    Solo incluye directorios cuyo nombre comience con 'barcode' (ej: barcode01).

    Parámetros:
        fastq_pass_dir: ruta al directorio fastq_pass.

    Retorna:
        Lista ordenada de rutas absolutas a los directorios de barcode.
    """
    barcodes = []
    for entry in sorted(os.listdir(fastq_pass_dir)):
        full_path = os.path.join(fastq_pass_dir, entry)
        if os.path.isdir(full_path) and entry.lower().startswith('barcode'):
            barcodes.append(full_path)
    return barcodes


def listar_fastq_gz(barcode_dir: str) -> list:
    """
    Lista todos los archivos .fastq.gz dentro de un directorio de barcode.

    Parámetros:
        barcode_dir: ruta al directorio del barcode.

    Retorna:
        Lista ordenada de rutas absolutas a archivos .fastq.gz.
    """
    archivos = []
    for entry in sorted(os.listdir(barcode_dir)):
        if entry.endswith('.fastq.gz'):
            archivos.append(os.path.join(barcode_dir, entry))
    return archivos


# =============================================================================
# Función principal de procesamiento
# =============================================================================

def procesar_barcode(barcode_dir: str, output_dir: str,
                     min_length: int, max_length: int,
                     min_quality: float) -> dict:
    """
    Procesa todas las lecturas de un barcode: filtra por longitud y calidad.

    Parámetros:
        barcode_dir: directorio con archivos .fastq.gz del barcode.
        output_dir:  directorio donde escribir el archivo filtrado.
        min_length:  longitud mínima permitida (bp).
        max_length:  longitud máxima permitida (bp).
        min_quality: Q-score medio mínimo permitido.

    Retorna:
        Diccionario con estadísticas del barcode:
            barcode, total_reads, passed_reads, failed_length, failed_quality,
            mean_length, median_length, mean_qscore, median_qscore
    """
    barcode_name = os.path.basename(barcode_dir)
    archivos_gz = listar_fastq_gz(barcode_dir)

    if not archivos_gz:
        print(f"  [AVISO] No se encontraron archivos .fastq.gz en {barcode_name}")
        return {
            'barcode': barcode_name,
            'total_reads': 0, 'passed_reads': 0,
            'failed_length': 0, 'failed_quality': 0,
            'mean_length': 0.0, 'median_length': 0.0,
            'mean_qscore': 0.0, 'median_qscore': 0.0
        }

    # Contadores
    total_reads = 0
    passed_reads = 0
    failed_length = 0
    failed_quality = 0

    # Listas para estadísticas de lecturas que PASARON el filtro
    passed_lengths = []
    passed_qscores = []

    # Archivo de salida para lecturas filtradas
    output_file = os.path.join(output_dir, f"{barcode_name}_filtered.fastq.gz")

    with gzip.open(output_file, 'wt', encoding='utf-8') as out_fh:
        for archivo in archivos_gz:
            for header, sequence, separator, quality in parse_fastq_gzip(archivo):
                total_reads += 1
                read_len = len(sequence)
                qscore = calcular_qscore_medio(quality)

                # --- Filtro de longitud ---
                if read_len < min_length or read_len > max_length:
                    failed_length += 1
                    continue

                # --- Filtro de calidad ---
                if qscore < min_quality:
                    failed_quality += 1
                    continue

                # --- La lectura pasó ambos filtros ---
                passed_reads += 1
                passed_lengths.append(read_len)
                passed_qscores.append(qscore)

                # Escribir al archivo de salida
                out_fh.write(f"{header}\n{sequence}\n{separator}\n{quality}\n")

    # Calcular estadísticas
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

    # Porcentaje de lecturas que pasaron
    pct = (passed_reads / total_reads * 100) if total_reads > 0 else 0.0

    print(f"  {barcode_name}: {total_reads:,} total → {passed_reads:,} pasaron "
          f"({pct:.1f}%) | descartadas: {failed_length:,} (longitud), "
          f"{failed_quality:,} (calidad)")

    return stats


def escribir_resumen_csv(stats_list: list, output_path: str):
    """
    Escribe el resumen estadístico de todos los barcodes en un archivo CSV.

    Parámetros:
        stats_list:  lista de diccionarios con estadísticas por barcode.
        output_path: ruta al archivo CSV de salida.
    """
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

    print(f"\n✓ Resumen CSV guardado en: {output_path}")


# =============================================================================
# Punto de entrada — CLI
# =============================================================================

def crear_parser() -> argparse.ArgumentParser:
    """
    Crea el parser de argumentos de línea de comandos.

    Retorna:
        ArgumentParser configurado con todos los parámetros del script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Preprocesamiento de lecturas FASTQ de secuenciación 16S rRNA (Oxford Nanopore).\n"
            "Filtra lecturas por longitud y calidad media (Phred Q-score)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplo de uso:\n"
            "  py preprocess_fastq.py --input C:\\ruta\\al\\Lote1 "
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
        help='Directorio de salida para los archivos filtrados y el resumen CSV.'
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
    """Función principal: parsea argumentos, ejecuta el pipeline de preprocesamiento."""
    parser = crear_parser()
    args = parser.parse_args()

    # --- Encabezado informativo ---
    print("=" * 72)
    print("  PREPROCESAMIENTO DE LECTURAS 16S rRNA — Oxford Nanopore")
    print("=" * 72)
    print(f"  Directorio de entrada : {args.input}")
    print(f"  Directorio de salida  : {args.output}")
    print(f"  Filtro de longitud    : {args.min_length} – {args.max_length} bp")
    print(f"  Filtro de calidad     : Q-score medio ≥ {args.min_quality}")
    print("=" * 72)

    # --- Localizar fastq_pass ---
    try:
        fastq_pass_dir = encontrar_directorio_fastq_pass(args.input)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n→ Directorio fastq_pass encontrado: {fastq_pass_dir}")

    # --- Obtener lista de barcodes ---
    barcodes = obtener_barcodes(fastq_pass_dir)
    if not barcodes:
        print("[ERROR] No se encontraron directorios de barcode en fastq_pass/.",
              file=sys.stderr)
        sys.exit(1)

    print(f"→ Barcodes encontrados: {len(barcodes)}")
    for bc in barcodes:
        print(f"    • {os.path.basename(bc)}")

    # --- Crear directorio de salida ---
    filtered_dir = os.path.join(args.output, 'filtered_reads')
    os.makedirs(filtered_dir, exist_ok=True)
    print(f"\n→ Lecturas filtradas se guardarán en: {filtered_dir}")

    # --- Procesar cada barcode ---
    print("\n" + "-" * 72)
    print("  Procesando barcodes...")
    print("-" * 72)

    inicio = time.time()
    all_stats = []

    for i, barcode_dir in enumerate(barcodes, 1):
        barcode_name = os.path.basename(barcode_dir)
        print(f"\n[{i}/{len(barcodes)}] Procesando {barcode_name}...")
        stats = procesar_barcode(
            barcode_dir=barcode_dir,
            output_dir=filtered_dir,
            min_length=args.min_length,
            max_length=args.max_length,
            min_quality=args.min_quality
        )
        all_stats.append(stats)

    elapsed = time.time() - inicio

    # --- Resumen global ---
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

    # --- Guardar CSV ---
    csv_path = os.path.join(args.output, 'preprocessing_summary.csv')
    escribir_resumen_csv(all_stats, csv_path)

    print(f"\n✓ Preprocesamiento completado exitosamente.")
    print(f"  Archivos filtrados en : {filtered_dir}")
    print(f"  Resumen CSV en        : {csv_path}")


if __name__ == '__main__':
    main()
