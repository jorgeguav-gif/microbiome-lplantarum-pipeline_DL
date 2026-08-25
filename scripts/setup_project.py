#!/usr/bin/env python3
"""
@author: Jorge Luis Gutiérrez-Ávila
@institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
@orcid: https://orcid.org/0000-0003-1630-954X
@github: jorgeguav-gif
"""

import os
import csv
from pathlib import Path

# === Directorio base del proyecto ===
BASE_DIR = Path(__file__).parent

# === 1. Crear estructura de directorios ===
DIRECTORIES = [
    "01_preprocessing",
    "02_quality_control",
    "03_classification",
    "04_statistics",
    "05_deep_learning",
    "results",
    "figures",
]

print("=" * 60)
print("CONFIGURACIÓN DEL PROYECTO DE ANÁLISIS 16S rRNA")
print("=" * 60)

print("\n[1/2] Creando directorios...")
for d in DIRECTORIES:
    dir_path = BASE_DIR / d
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {dir_path}")

# === 2. Generar plantilla de metadatos ===
print("\n[2/2] Generating plantilla de metadatos...")

batchs = [
    ("Batch1", 24),
    ("Batch2", 24),
    ("Batch3", 16),
]

COLUMNS = [
    "sample_id",
    "barcode",
    "batch",
    "sexo",
    "tratamiento",
    "cepa_probiotico",
    "peso_raton_g",
    "edad_semanas",
    "jaula",
    "fecha_sacrificio",
    "notas",
]

rows = []
for batch_name, n_barcodes in batchs:
    batch_prefix = batch_name.replace("Batch", "L")  # L1, L2, L3
    for i in range(1, n_barcodes + 1):
        sample_id = f"{batch_prefix}_BC{i:02d}"
        barcode = f"barcode{i:02d}"
        row = {
            "sample_id": sample_id,
            "barcode": barcode,
            "batch": batch_name,
            "sexo": "",
            "tratamiento": "",
            "cepa_probiotico": "",
            "peso_raton_g": "",
            "edad_semanas": "",
            "jaula": "",
            "fecha_sacrificio": "",
            "notas": "",
        }
        rows.append(row)

csv_path = BASE_DIR / "metadata_template.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    f.write("# ============================================================\n")
    f.write("# PLANTILLA DE METADATOS - Análisis 16S rRNA Ratones CD1\n")
    f.write("# ============================================================\n")
    f.write("# Proyecto: Efecto de cepas de P. plantarum en microbiota intestinal\n")
    f.write("# Investigador: Jorge\n")
    f.write(f"# Fecha de creación: 2026-07-16\n")
    f.write("# \n")
    f.write("# INSTRUCCIONES:\n")
    f.write("#   1. Completar las columnas vacías con los datos experimentales reales\n")
    f.write("#   2. sexo: M (macho) o F (hembra)\n")
    f.write("#   3. tratamiento: Control, LM20, G7 o P128\n")
    f.write("#   4. cepa_probiotico: NA (para Control), P_plantarum_LM20,\n")
    f.write("#      P_plantarum_G7 o P_plantarum_P128\n")
    f.write("#   5. peso_raton_g: peso en gramos (ej: 28.5)\n")
    f.write("#   6. edad_semanas: edad en semanas al momento del sacrificio\n")
    f.write("#   7. jaula: identificador de jaula\n")
    f.write("#   8. fecha_sacrificio: formato YYYY-MM-DD\n")
    f.write("#   9. notas: observaciones adicionales\n")
    f.write("# \n")
    f.write("# Distribución de barcodes por batch:\n")
    f.write("#   Batch1: barcode01-barcode24 (24 samples)\n")
    f.write("#   Batch2: barcode01-barcode24 (24 samples)\n")
    f.write("#   Batch3: barcode01-barcode16 (16 samples)\n")
    f.write("#   Total: 64 samples\n")
    f.write("# \n")
    f.write("# Diseño experimental: 64 ratones CD1 (32M + 32F)\n")
    f.write("#   4 tratamientos x 16 ratones = 64 total\n")
    f.write("#   (8M + 8F por tratamiento)\n")
    f.write("# ============================================================\n")

    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(f"  ✓ {csv_path}")
print(f"    → {len(rows)} samples registradas ({', '.join(f'{name}: {n}' for name, n in batchs)})")

# === Resumen ===
print("\n" + "=" * 60)
print("ESTRUCTURA DEL PROYECTO:")
print("=" * 60)
for d in DIRECTORIES:
    print(f"  📁 analysis/{d}/")
print(f"  📄 analysis/metadata_template.csv ({len(rows)} samples)")
print("\n✅ Project configured correctly.")
print("\n⚠️  SIGUIENTE PASO: Completar metadata_template.csv con los")
print("   datos experimentales reales (sexo, tratamiento, etc.)")
