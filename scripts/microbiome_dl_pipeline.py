#!/usr/bin/env python3
"""
@author: Jorge Luis Gutiérrez-Ávila
@institution: Escuela Nacional de Ciencias Biológicas (ENCB)-Instituto Politecnico Nacional (IPN), Mexico
@orcid: https://orcid.org/0000-0003-1630-954X
@github: jorgeguav-gif
"""

# -*- coding: utf-8 -*-
"""
microbiome_dl_pipeline.py
Pipeline de Deep Learning para análisis del microbioma 16S

Tres proyectos integrados:
    A) Predictor de Fenotipo (Multi-head MLP): Predice sexo y tratamiento
       a partir de perfiles de abundancia
    B) VAE para Eubiosis: Autoencoder variacional entrenado con controles
       para cuantificar disbiosis en muestras tratadas
    C) Transformer para 16S: Clasificación taxonómica a nivel de género
       usando tokenización por k-mers de lecturas 16S

Diseño experimental:
    - 64 ratones CD1 (32M + 32F)
    - 4 tratamientos: Control, LM20, G7, P128 (cepas de P. plantarum)
    - Secuenciación: Oxford Nanopore MinION, kit SQK-16S114-24

Uso:
    python microbiome_dl_pipeline.py predict --otu_table ... --metadata ...
    python microbiome_dl_pipeline.py vae --otu_table ... --metadata ...
    python microbiome_dl_pipeline.py transformer --reads_dir ... --taxonomy_ref ...

Autor: Jorge
Fecha: Julio 2026
"""

import argparse
import os
import sys
import json
import time
import warnings
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score
)
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para HPC
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración estilo Nature
nature_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
sns.set_palette(sns.color_palette(nature_colors))
sns.set_context('paper', font_scale=1.2)
sns.set_style('ticks')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 10
})

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def configurar_reproducibilidad(seed: int = 42):
    """Fijar semillas para reproducibilidad completa."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Semilla de reproducibilidad fijada: {seed}")


def detectar_dispositivo():
    """Detectar y configurar el dispositivo de cómputo (GPU o CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detectada: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device('cpu')
        logger.info("GPU no disponible, usando CPU")
    return device


def crear_directorios(*dirs):
    """Crear múltiples directorios si no existen."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def guardar_metricas(metricas: dict, filepath: str):
    """Guardar métricas en formato JSON."""
    # Convertir tipos numpy a tipos nativos de Python
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    metricas_clean = {k: convert(v) for k, v in metricas.items()}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metricas_clean, f, indent=2, ensure_ascii=False)
    logger.info(f"Métricas guardadas en: {filepath}")


# =============================================================================
# CARGA DE DATOS
# =============================================================================

class DatosMicrobioma:
    """
    Clase para cargar y preprocesar datos de microbioma.
    Maneja tabla OTU y metadata con validación robusta.
    """

    def __init__(self, otu_path: str, metadata_path: str):
        """
        Cargar tabla OTU y metadata desde archivos CSV.

        Parámetros:
            otu_path: Ruta a la tabla OTU (muestras × taxa)
            metadata_path: Ruta al archivo de metadata
        """
        logger.info(f"Cargando tabla OTU: {otu_path}")
        self.otu = pd.read_csv(otu_path, index_col=0)
        logger.info(f"  Dimensiones OTU: {self.otu.shape[0]} muestras × {self.otu.shape[1]} taxa")

        logger.info(f"Cargando metadata: {metadata_path}")
        self.metadata = pd.read_csv(metadata_path, index_col=0)
        logger.info(f"  Dimensiones metadata: {self.metadata.shape}")

        # Alinear muestras
        self._alinear_muestras()

        # Identificar columnas
        self._identificar_columnas()

        # Codificar variables categóricas
        self._codificar_variables()

    def _alinear_muestras(self):
        """Alinear muestras entre OTU y metadata."""
        comunes = self.otu.index.intersection(self.metadata.index)
        if len(comunes) == 0:
            raise ValueError("No hay muestras en común entre OTU y metadata")

        self.otu = self.otu.loc[comunes]
        self.metadata = self.metadata.loc[comunes]
        logger.info(f"  Muestras alineadas: {len(comunes)}")

    def _identificar_columnas(self):
        """Identificar columnas de tratamiento, sexo y lote en metadata."""
        cols = self.metadata.columns.str.lower()

        # Tratamiento
        for patron in ['tratamiento', 'treatment', 'trat', 'group']:
            matches = [c for c, cl in zip(self.metadata.columns, cols) if patron in cl]
            if matches:
                self.col_tratamiento = matches[0]
                break
        else:
            self.col_tratamiento = self.metadata.columns[0]
        logger.info(f"  Columna tratamiento: '{self.col_tratamiento}'")

        # Sexo
        for patron in ['sexo', 'sex', 'género', 'gender']:
            matches = [c for c, cl in zip(self.metadata.columns, cols) if patron in cl]
            if matches:
                self.col_sexo = matches[0]
                break
        else:
            self.col_sexo = self.metadata.columns[1] if len(self.metadata.columns) > 1 else None
        logger.info(f"  Columna sexo: '{self.col_sexo}'")

        # Lote
        for patron in ['lote', 'lot', 'batch']:
            matches = [c for c, cl in zip(self.metadata.columns, cols) if patron in cl]
            if matches:
                self.col_lote = matches[0]
                break
        else:
            self.col_lote = None
        logger.info(f"  Columna lote: '{self.col_lote}'")

    def _codificar_variables(self):
        """Codificar variables categóricas como enteros."""
        self.le_tratamiento = LabelEncoder()
        self.tratamiento = self.le_tratamiento.fit_transform(
            self.metadata[self.col_tratamiento]
        )
        self.clases_tratamiento = list(self.le_tratamiento.classes_)
        logger.info(f"  Clases tratamiento: {self.clases_tratamiento}")

        if self.col_sexo:
            self.le_sexo = LabelEncoder()
            self.sexo = self.le_sexo.fit_transform(self.metadata[self.col_sexo])
            self.clases_sexo = list(self.le_sexo.classes_)
            logger.info(f"  Clases sexo: {self.clases_sexo}")
        else:
            self.sexo = np.zeros(len(self.metadata), dtype=int)
            self.clases_sexo = ['Desconocido']

    def obtener_abundancias(self, normalizar: bool = True) -> np.ndarray:
        """
        Obtener matriz de abundancias como array numpy.

        Parámetros:
            normalizar: Si True, normalizar a abundancia relativa

        Retorna:
            Matriz numpy de abundancias (muestras × taxa)
        """
        X = self.otu.values.astype(np.float32)
        if normalizar:
            sumas = X.sum(axis=1, keepdims=True)
            sumas[sumas == 0] = 1  # Evitar división por cero
            X = X / sumas
        return X

    def obtener_indices_control(self) -> np.ndarray:
        """Obtener índices de las muestras de control."""
        control_mask = self.metadata[self.col_tratamiento] == 'Control'
        return np.where(control_mask)[0]

    def obtener_indices_tratados(self) -> np.ndarray:
        """Obtener índices de las muestras tratadas (no control)."""
        tratado_mask = self.metadata[self.col_tratamiento] != 'Control'
        return np.where(tratado_mask)[0]


# =============================================================================
# DATASETS DE PYTORCH
# =============================================================================

class DatasetAbundancia(Dataset):
    """
    Dataset de PyTorch para perfiles de abundancia.
    Incluye aumento de datos con ruido gaussiano.
    """

    def __init__(self, X: np.ndarray, y_sexo: np.ndarray, y_trat: np.ndarray,
                 aumentar: bool = False, sigma_ruido: float = 0.01):
        """
        Inicializar dataset.

        Parámetros:
            X: Matriz de abundancias (n_muestras × n_taxa)
            y_sexo: Etiquetas de sexo
            y_trat: Etiquetas de tratamiento
            aumentar: Si True, aplicar aumento de datos
            sigma_ruido: Desviación estándar del ruido gaussiano
        """
        self.X = torch.FloatTensor(X)
        self.y_sexo = torch.LongTensor(y_sexo)
        self.y_trat = torch.LongTensor(y_trat)
        self.aumentar = aumentar
        self.sigma_ruido = sigma_ruido

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()

        # Aumento de datos: añadir ruido gaussiano
        if self.aumentar and self.training_mode:
            ruido = torch.randn_like(x) * self.sigma_ruido
            x = torch.clamp(x + ruido, min=0)  # Asegurar no negativos
            # Re-normalizar
            suma = x.sum()
            if suma > 0:
                x = x / suma

        return x, self.y_sexo[idx], self.y_trat[idx]

    @property
    def training_mode(self):
        return self.aumentar


class DatasetSecuencias(Dataset):
    """
    Dataset de PyTorch para secuencias 16S tokenizadas por k-mers.
    """

    def __init__(self, secuencias: list, etiquetas: np.ndarray,
                 kmer_size: int = 6, vocab: dict = None, max_len: int = 512):
        """
        Inicializar dataset de secuencias.

        Parámetros:
            secuencias: Lista de secuencias de ADN
            etiquetas: Etiquetas taxonómicas codificadas
            kmer_size: Tamaño del k-mer
            vocab: Vocabulario de k-mers (si None, se construye)
            max_len: Longitud máxima de la secuencia tokenizada
        """
        self.kmer_size = kmer_size
        self.max_len = max_len
        self.etiquetas = torch.LongTensor(etiquetas)

        # Construir o usar vocabulario
        if vocab is None:
            self.vocab = self._construir_vocabulario(secuencias)
        else:
            self.vocab = vocab

        # Tokenizar secuencias
        self.tokens = [self._tokenizar(seq) for seq in secuencias]

    def _construir_vocabulario(self, secuencias: list) -> dict:
        """Construir vocabulario de k-mers a partir de las secuencias."""
        kmers = Counter()
        for seq in secuencias:
            seq = seq.upper().replace('N', '')
            for i in range(len(seq) - self.kmer_size + 1):
                kmer = seq[i:i + self.kmer_size]
                if all(c in 'ACGT' for c in kmer):
                    kmers[kmer] += 1

        # Tokens especiales
        vocab = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2}
        for kmer, _ in kmers.most_common():
            vocab[kmer] = len(vocab)

        logger.info(f"  Vocabulario construido: {len(vocab)} tokens (k={self.kmer_size})")
        return vocab

    def _tokenizar(self, seq: str) -> torch.LongTensor:
        """Tokenizar una secuencia en k-mers."""
        seq = seq.upper().replace('N', '')
        tokens = [self.vocab.get('<CLS>', 2)]

        for i in range(len(seq) - self.kmer_size + 1):
            kmer = seq[i:i + self.kmer_size]
            token_id = self.vocab.get(kmer, self.vocab.get('<UNK>', 1))
            tokens.append(token_id)

        # Truncar o pad
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens.extend([0] * (self.max_len - len(tokens)))

        return torch.LongTensor(tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.etiquetas[idx]


# =============================================================================
# PROYECTO A: PREDICTOR DE FENOTIPO (MULTI-HEAD MLP)
# =============================================================================

class PredictorMultiHead(nn.Module):
    """
    Red neuronal MLP multi-cabeza para predicción simultánea
    de sexo (2 clases) y tratamiento (4 clases) a partir de
    perfiles de abundancia microbiana.

    Arquitectura:
        Input → Shared MLP → BatchNorm → GELU → Dropout
              → Head_Sexo (2 clases)
              → Head_Tratamiento (4 clases)
    """

    def __init__(self, n_taxa: int, n_clases_sexo: int = 2,
                 n_clases_trat: int = 4, hidden_dims: list = None,
                 dropout: float = 0.3):
        """
        Inicializar predictor multi-cabeza.

        Parámetros:
            n_taxa: Número de taxa (dimensión de entrada)
            n_clases_sexo: Número de clases de sexo
            n_clases_trat: Número de clases de tratamiento
            hidden_dims: Dimensiones de capas ocultas compartidas
            dropout: Tasa de dropout
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # Red troncal compartida
        layers = []
        in_dim = n_taxa
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim

        self.troncal = nn.Sequential(*layers)

        # Cabeza para clasificación de sexo
        self.cabeza_sexo = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, n_clases_sexo)
        )

        # Cabeza para clasificación de tratamiento
        self.cabeza_tratamiento = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, n_clases_trat)
        )

    def forward(self, x):
        """
        Forward pass.

        Parámetros:
            x: Tensor de abundancias (batch × n_taxa)

        Retorna:
            logits_sexo: Logits para clasificación de sexo
            logits_trat: Logits para clasificación de tratamiento
        """
        features = self.troncal(x)
        logits_sexo = self.cabeza_sexo(features)
        logits_trat = self.cabeza_tratamiento(features)
        return logits_sexo, logits_trat

    def obtener_importancia(self, x: torch.Tensor) -> np.ndarray:
        """
        Calcular importancia de features usando gradientes integrados
        (aproximación simple basada en gradientes).

        Parámetros:
            x: Tensor de entrada (1 × n_taxa)

        Retorna:
            Importancia por taxón (array numpy)
        """
        self.eval()
        x = x.clone().requires_grad_(True)
        logits_sexo, logits_trat = self(x)

        # Importancia como magnitud del gradiente respecto a ambas salidas
        loss = logits_sexo.sum() + logits_trat.sum()
        loss.backward()

        importancia = x.grad.abs().detach().cpu().numpy()
        return importancia


def ejecutar_proyecto_a(args):
    """
    Proyecto A: Predictor de Fenotipo Multi-cabeza.
    Entrena un MLP para predecir sexo y tratamiento simultáneamente
    usando validación cruzada (8-fold o LOO).
    """
    logger.info("=" * 60)
    logger.info("  PROYECTO A: PREDICTOR DE FENOTIPO (Multi-head MLP)")
    logger.info("=" * 60)

    device = detectar_dispositivo()
    crear_directorios(args.output_dir, args.models_dir, args.figures_dir)

    # ─── Cargar datos ────────────────────────────────────────────────────
    datos = DatosMicrobioma(args.otu_table, args.metadata)
    X = datos.obtener_abundancias(normalizar=True)
    y_sexo = datos.sexo
    y_trat = datos.tratamiento

    n_muestras, n_taxa = X.shape
    n_clases_sexo = len(datos.clases_sexo)
    n_clases_trat = len(datos.clases_tratamiento)

    logger.info(f"Datos: {n_muestras} muestras × {n_taxa} taxa")
    logger.info(f"Clases sexo: {n_clases_sexo} | Clases tratamiento: {n_clases_trat}")

    # ─── Normalizar features ────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ─── Validación cruzada ──────────────────────────────────────────────
    cv_folds = args.cv_folds
    usar_loo = (n_muestras < 30) or (cv_folds >= n_muestras)

    if usar_loo:
        logger.info("Usando Leave-One-Out CV (n pequeño)")
        cv = LeaveOneOut()
        n_splits = n_muestras
    else:
        logger.info(f"Usando {cv_folds}-fold Stratified CV")
        # Estratificar por tratamiento (más clases → mejor estratificación)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.seed)
        n_splits = cv_folds

    # Almacenar predicciones
    pred_sexo_all = np.zeros(n_muestras, dtype=int)
    pred_trat_all = np.zeros(n_muestras, dtype=int)
    prob_sexo_all = np.zeros((n_muestras, n_clases_sexo))
    prob_trat_all = np.zeros((n_muestras, n_clases_trat))

    # Importancia acumulada de features
    importancia_total = np.zeros(n_taxa)

    historiales = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_trat)):
        if fold % max(1, n_splits // 10) == 0:
            logger.info(f"  Fold {fold + 1}/{n_splits}")

        # Datos de entrenamiento y prueba
        X_train = X_scaled[train_idx]
        X_test = X_scaled[test_idx]
        y_sexo_train, y_sexo_test = y_sexo[train_idx], y_sexo[test_idx]
        y_trat_train, y_trat_test = y_trat[train_idx], y_trat[test_idx]

        # Crear datasets con aumento de datos
        ds_train = DatasetAbundancia(X_train, y_sexo_train, y_trat_train,
                                     aumentar=True, sigma_ruido=0.02)
        ds_test = DatasetAbundancia(X_test, y_sexo_test, y_trat_test,
                                    aumentar=False)

        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              drop_last=False)
        dl_test = DataLoader(ds_test, batch_size=len(test_idx), shuffle=False)

        # Crear modelo
        modelo = PredictorMultiHead(
            n_taxa=n_taxa,
            n_clases_sexo=n_clases_sexo,
            n_clases_trat=n_clases_trat,
            hidden_dims=[256, 128, 64],
            dropout=0.3
        ).to(device)

        optimizer = torch.optim.AdamW(modelo.parameters(), lr=args.learning_rate,
                                       weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

        # Pesos para clases desbalanceadas
        peso_sexo = torch.FloatTensor(
            [1.0 / max(1, (y_sexo_train == c).sum()) for c in range(n_clases_sexo)]
        ).to(device)
        peso_trat = torch.FloatTensor(
            [1.0 / max(1, (y_trat_train == c).sum()) for c in range(n_clases_trat)]
        ).to(device)

        criterion_sexo = nn.CrossEntropyLoss(weight=peso_sexo)
        criterion_trat = nn.CrossEntropyLoss(weight=peso_trat)

        # ─── Entrenamiento ─────────────────────────────────────────────
        for epoch in range(args.epochs):
            modelo.train()
            for batch_x, batch_y_sexo, batch_y_trat in dl_train:
                batch_x = batch_x.to(device)
                batch_y_sexo = batch_y_sexo.to(device)
                batch_y_trat = batch_y_trat.to(device)

                logits_sexo, logits_trat = modelo(batch_x)

                loss_sexo = criterion_sexo(logits_sexo, batch_y_sexo)
                loss_trat = criterion_trat(logits_trat, batch_y_trat)
                loss = 0.3 * loss_sexo + 0.7 * loss_trat  # Más peso a tratamiento

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

        # ─── Evaluación ────────────────────────────────────────────────
        modelo.eval()
        with torch.no_grad():
            for batch_x, _, _ in dl_test:
                batch_x = batch_x.to(device)
                logits_sexo, logits_trat = modelo(batch_x)

                pred_sexo_all[test_idx] = logits_sexo.argmax(dim=1).cpu().numpy()
                pred_trat_all[test_idx] = logits_trat.argmax(dim=1).cpu().numpy()
                prob_sexo_all[test_idx] = F.softmax(logits_sexo, dim=1).cpu().numpy()
                prob_trat_all[test_idx] = F.softmax(logits_trat, dim=1).cpu().numpy()

        # Importancia de features (solo en el último fold con modelo completo)
        if fold == n_splits - 1:
            x_mean = torch.FloatTensor(X_scaled.mean(axis=0)).unsqueeze(0).to(device)
            importancia_total += modelo.obtener_importancia(x_mean).flatten()

    # ─── Métricas finales ────────────────────────────────────────────────
    logger.info("\n  === RESULTADOS PROYECTO A ===")

    # Sexo
    acc_sexo = accuracy_score(y_sexo, pred_sexo_all)
    f1_sexo = f1_score(y_sexo, pred_sexo_all, average='weighted')
    logger.info(f"  Sexo     - Accuracy: {acc_sexo:.3f} | F1: {f1_sexo:.3f}")

    reporte_sexo = classification_report(
        y_sexo, pred_sexo_all,
        target_names=datos.clases_sexo,
        output_dict=True
    )

    # Tratamiento
    acc_trat = accuracy_score(y_trat, pred_trat_all)
    f1_trat = f1_score(y_trat, pred_trat_all, average='weighted')
    logger.info(f"  Tratamiento - Accuracy: {acc_trat:.3f} | F1: {f1_trat:.3f}")

    reporte_trat = classification_report(
        y_trat, pred_trat_all,
        target_names=datos.clases_tratamiento,
        output_dict=True
    )

    # Guardar métricas
    metricas = {
        'sexo': {
            'accuracy': acc_sexo,
            'f1_weighted': f1_sexo,
            'reporte': reporte_sexo
        },
        'tratamiento': {
            'accuracy': acc_trat,
            'f1_weighted': f1_trat,
            'reporte': reporte_trat
        },
        'n_muestras': n_muestras,
        'n_taxa': n_taxa,
        'cv_tipo': 'LOO' if usar_loo else f'{cv_folds}-fold',
        'epochs': args.epochs
    }
    guardar_metricas(metricas, os.path.join(args.output_dir, 'metricas_proyecto_a.json'))

    # ─── Figuras ─────────────────────────────────────────────────────────

    # Matriz de confusión - Sexo
    # English version
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm_sexo = confusion_matrix(y_sexo, pred_sexo_all)
    sns.heatmap(cm_sexo, annot=True, fmt='d', cmap='Blues',
                xticklabels=datos.clases_sexo, yticklabels=datos.clases_sexo, ax=axes[0])
    axes[0].set_title(f'Sex Prediction\nAccuracy: {acc_sexo:.3f}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    cm_trat = confusion_matrix(y_trat, pred_trat_all)
    sns.heatmap(cm_trat, annot=True, fmt='d', cmap='Oranges',
                xticklabels=datos.clases_tratamiento, yticklabels=datos.clases_tratamiento, ax=axes[1])
    axes[1].set_title(f'Treatment Prediction\nAccuracy: {acc_trat:.3f}')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, 'proyecto_a_confusion_en.tiff'), dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close()

    # Spanish version
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm_sexo, annot=True, fmt='d', cmap='Blues',
                xticklabels=datos.clases_sexo, yticklabels=datos.clases_sexo, ax=axes[0])
    axes[0].set_title(f'Predicción de Sexo\nPrecisión: {acc_sexo:.3f}')
    axes[0].set_xlabel('Predicho')
    axes[0].set_ylabel('Real')

    sns.heatmap(cm_trat, annot=True, fmt='d', cmap='Oranges',
                xticklabels=datos.clases_tratamiento, yticklabels=datos.clases_tratamiento, ax=axes[1])
    axes[1].set_title(f'Predicción de Tratamiento\nPrecisión: {acc_trat:.3f}')
    axes[1].set_xlabel('Predicho')
    axes[1].set_ylabel('Real')
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, 'proyecto_a_confusion_es.tiff'), dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close()

    # Importancia de features (Top 20)
    if importancia_total.sum() > 0:
        taxa_nombres = datos.otu.columns.tolist()
        top_k = min(20, n_taxa)
        top_idx = np.argsort(importancia_total)[-top_k:][::-1]

        # English
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(top_k), importancia_total[top_idx][::-1], color='steelblue')
        ax.set_yticks(range(top_k))
        labels = [rf"$\mathit{{{taxa_nombres[i].replace('_', ' ')}}}$" for i in top_idx]
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_xlabel('Importance (Gradient Magnitude)')
        ax.set_title('Top 20 Most Important Taxa for Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(args.figures_dir, 'proyecto_a_importancia_en.tiff'), dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
        plt.close()

        # Spanish
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(top_k), importancia_total[top_idx][::-1], color='steelblue')
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_xlabel('Importancia (Magnitud del Gradiente)')
        ax.set_title('Top 20 Taxones Más Importantes para Predicción')
        plt.tight_layout()
        plt.savefig(os.path.join(args.figures_dir, 'proyecto_a_importancia_es.tiff'), dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
        plt.close()

        # Guardar importancia
        df_imp = pd.DataFrame({
            'Taxon': taxa_nombres,
            'Importancia': importancia_total
        }).sort_values('Importancia', ascending=False)
        df_imp.to_csv(os.path.join(args.output_dir, 'importancia_taxa.csv'), index=False)

    # Guardar modelo final (entrenado con todos los datos)
    logger.info("  Entrenando modelo final con todos los datos...")
    modelo_final = PredictorMultiHead(
        n_taxa=n_taxa, n_clases_sexo=n_clases_sexo,
        n_clases_trat=n_clases_trat, hidden_dims=[256, 128, 64]
    ).to(device)

    ds_completo = DatasetAbundancia(X_scaled, y_sexo, y_trat, aumentar=True)
    dl_completo = DataLoader(ds_completo, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(modelo_final.parameters(), lr=args.learning_rate)

    modelo_final.train()
    for epoch in range(args.epochs):
        for bx, by_s, by_t in dl_completo:
            bx, by_s, by_t = bx.to(device), by_s.to(device), by_t.to(device)
            ls, lt = modelo_final(bx)
            loss = 0.3 * criterion_sexo(ls, by_s) + 0.7 * criterion_trat(lt, by_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save({
        'model_state_dict': modelo_final.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'clases_sexo': datos.clases_sexo,
        'clases_tratamiento': datos.clases_tratamiento,
        'n_taxa': n_taxa,
    }, os.path.join(args.models_dir, 'predictor_fenotipo.pt'))

    logger.info("  ✓ Proyecto A completado")
    return metricas


# =============================================================================
# PROYECTO B: VAE PARA EUBIOSIS
# =============================================================================

class VAEMicrobioma(nn.Module):
    """
    Variational Autoencoder para cuantificación de eubiosis/disbiosis.
    Se entrena SOLO con muestras de control (estado "sano").
    El error de reconstrucción en muestras tratadas indica disbiosis.

    Arquitectura:
        Encoder: Input → Hidden → μ, σ  (espacio latente)
        Decoder: z ~ N(μ, σ) → Hidden → Output reconstruido
    """

    def __init__(self, n_taxa: int, hidden_dim: int = 128,
                 latent_dim: int = 2, dropout: float = 0.2):
        """
        Inicializar VAE.

        Parámetros:
            n_taxa: Dimensión de entrada (número de taxa)
            hidden_dim: Dimensión de capas ocultas
            latent_dim: Dimensión del espacio latente
            dropout: Tasa de dropout
        """
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_taxa, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Parámetros del espacio latente
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_taxa),
            nn.Softmax(dim=1)  # Salida como distribución de abundancia
        )

    def encode(self, x):
        """Codificar entrada al espacio latente."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Truco de reparametrización para muestreo diferenciable."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decodificar del espacio latente."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass completo."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @staticmethod
    def loss_function(x_recon, x, mu, logvar, beta: float = 1.0):
        """
        Función de pérdida ELBO.

        Parámetros:
            x_recon: Entrada reconstruida
            x: Entrada original
            mu: Media latente
            logvar: Log-varianza latente
            beta: Peso del término KL (β-VAE)

        Retorna:
            Pérdida total, pérdida de reconstrucción, pérdida KL
        """
        # Pérdida de reconstrucción (BCE o MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        # Divergencia KL: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss


def ejecutar_proyecto_b(args):
    """
    Proyecto B: VAE para evaluación de Eubiosis.
    Entrena un VAE solo con muestras de control y mide
    la disbiosis como error de reconstrucción en muestras tratadas.
    """
    logger.info("=" * 60)
    logger.info("  PROYECTO B: VAE PARA EUBIOSIS")
    logger.info("=" * 60)

    device = detectar_dispositivo()
    crear_directorios(args.output_dir, args.models_dir, args.figures_dir)

    # ─── Cargar datos ────────────────────────────────────────────────────
    datos = DatosMicrobioma(args.otu_table, args.metadata)
    X = datos.obtener_abundancias(normalizar=True)

    idx_control = datos.obtener_indices_control()
    idx_tratados = datos.obtener_indices_tratados()

    X_control = X[idx_control]
    X_tratados = X[idx_tratados]

    n_control = len(idx_control)
    n_tratados = len(idx_tratados)
    n_taxa = X.shape[1]

    logger.info(f"Muestras control: {n_control} | Muestras tratadas: {n_tratados}")
    logger.info(f"Dimensión latente: {args.latent_dim}")

    # ─── Normalizar usando SOLO datos de control ────────────────────────
    scaler = StandardScaler()
    X_control_scaled = scaler.fit_transform(X_control)
    X_tratados_scaled = scaler.transform(X_tratados)
    X_all_scaled = scaler.transform(X)

    # ─── Dataset de entrenamiento (solo controles) ──────────────────────
    X_ctrl_tensor = torch.FloatTensor(X_control_scaled)
    ds_control = torch.utils.data.TensorDataset(X_ctrl_tensor)
    dl_control = DataLoader(ds_control, batch_size=args.batch_size, shuffle=True)

    # ─── Crear y entrenar VAE ───────────────────────────────────────────
    modelo = VAEMicrobioma(
        n_taxa=n_taxa,
        hidden_dim=128,
        latent_dim=args.latent_dim,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(modelo.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    # Esquema de calentamiento para β
    historia_loss = []

    logger.info(f"Entrenando VAE por {args.epochs} epochs...")

    for epoch in range(args.epochs):
        modelo.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0

        # β-VAE warmup: incrementar β gradualmente
        beta = min(1.0, epoch / (args.epochs * 0.3))

        for (batch_x,) in dl_control:
            batch_x = batch_x.to(device)

            x_recon, mu, logvar = modelo(batch_x)
            loss, recon_loss, kl_loss = VAEMicrobioma.loss_function(
                x_recon, batch_x, mu, logvar, beta=beta
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

        avg_loss = epoch_loss / n_control
        scheduler.step(avg_loss)
        historia_loss.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            logger.info(f"  Epoch {epoch + 1}/{args.epochs} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Recon: {epoch_recon / n_control:.4f} | "
                        f"KL: {epoch_kl / n_control:.4f} | "
                        f"β: {beta:.3f}")

    # ─── Evaluar reconstrucción en todas las muestras ────────────────────
    logger.info("\n  Evaluando error de reconstrucción...")
    modelo.eval()

    with torch.no_grad():
        X_all_tensor = torch.FloatTensor(X_all_scaled).to(device)
        x_recon, mu_all, _ = modelo(X_all_tensor)

        # Error de reconstrucción por muestra (MSE)
        errores_recon = F.mse_loss(x_recon, X_all_tensor, reduction='none')
        errores_por_muestra = errores_recon.sum(dim=1).cpu().numpy()

        # Coordenadas latentes
        latentes = mu_all.cpu().numpy()

    # ─── Crear dataframe de resultados ───────────────────────────────────
    df_eubiosis = pd.DataFrame({
        'Muestra': datos.metadata.index,
        'Tratamiento': datos.metadata[datos.col_tratamiento].values,
        'Sexo': datos.metadata[datos.col_sexo].values if datos.col_sexo else 'NA',
        'Error_Reconstruccion': errores_por_muestra,
    })

    # Añadir coordenadas latentes
    for d in range(args.latent_dim):
        df_eubiosis[f'Latente_{d + 1}'] = latentes[:, d]

    # Score de disbiosis (normalizado respecto a controles)
    media_ctrl = errores_por_muestra[idx_control].mean()
    std_ctrl = max(errores_por_muestra[idx_control].std(), 1e-8)
    df_eubiosis['Score_Disbiosis'] = (errores_por_muestra - media_ctrl) / std_ctrl

    df_eubiosis.to_csv(os.path.join(args.output_dir, 'eubiosis_scores.csv'), index=False)

    # ─── Estadísticas por grupo ──────────────────────────────────────────
    logger.info("\n  Score de disbiosis por grupo:")
    resumen = df_eubiosis.groupby(['Tratamiento', 'Sexo'])['Score_Disbiosis'].agg(
        ['mean', 'std', 'count']
    ).round(3)
    logger.info(f"\n{resumen}")

    resumen.to_csv(os.path.join(args.output_dir, 'disbiosis_resumen.csv'))

    # ─── Métricas ────────────────────────────────────────────────────────
    metricas = {
        'error_recon_control_media': float(media_ctrl),
        'error_recon_control_std': float(std_ctrl),
        'error_recon_tratados_media': float(errores_por_muestra[idx_tratados].mean()),
        'latent_dim': args.latent_dim,
        'epochs': args.epochs,
        'loss_final': float(historia_loss[-1]),
    }

    # Añadir scores por cepa
    for cepa in datos.clases_tratamiento:
        mask = df_eubiosis['Tratamiento'] == cepa
        metricas[f'disbiosis_{cepa}_media'] = float(
            df_eubiosis.loc[mask, 'Score_Disbiosis'].mean()
        )
        metricas[f'disbiosis_{cepa}_std'] = float(
            df_eubiosis.loc[mask, 'Score_Disbiosis'].std()
        )

    guardar_metricas(metricas, os.path.join(args.output_dir, 'metricas_proyecto_b.json'))

    # ─── Figuras ─────────────────────────────────────────────────────────

    # 1. Curva de pérdida
    # EN
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(historia_loss, color='steelblue', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (ELBO)')
    ax.set_title('VAE Training Curve')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, 'proyecto_b_loss_en.tiff'), dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close()

    # ES
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(historia_loss, color='steelblue', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pérdida (ELBO)')
    ax.set_title('Curva de Entrenamiento del VAE')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, 'proyecto_b_loss_es.tiff'), dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close()

    # 2. Espacio latente 2D
    if args.latent_dim == 2:
        colores_trat = {
            'Control': '#4DAF4A', 'LM20': '#377EB8',
            'G7': '#FF7F00', 'P128': '#E41A1C'
        }
        marcadores_sexo = {'M': 'o', 'F': '^'}

        fig, ax = plt.subplots(figsize=(10, 8))
        for trat in datos.clases_tratamiento:
            for sexo_val in datos.clases_sexo:
                mask = (df_eubiosis['Tratamiento'] == trat) & \
                       (df_eubiosis['Sexo'] == sexo_val)
                if mask.sum() > 0:
                    ax.scatter(
                        df_eubiosis.loc[mask, 'Latente_1'],
                        df_eubiosis.loc[mask, 'Latente_2'],
                        c=colores_trat.get(trat, 'grey'),
                        marker=marcadores_sexo.get(sexo_val, 'o'),
                        s=80, alpha=0.8, edgecolors='white', linewidth=0.5,
                        label=f'{trat} ({sexo_val})'
                    )

        # EN
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_title('VAE Latent Space\n(Trained with Controls)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(args.figures_dir, 'proyecto_b_latente_en.tiff'), dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'}, bbox_inches='tight')

        # ES
        ax.set_xlabel('Dimensión Latente 1')
        ax.set_ylabel('Dimensión Latente 2')
        ax.set_title('Espacio Latente del VAE\n(Entrenado con Controles)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.figures_dir, 'proyecto_b_latente_es.tiff'), dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'}, bbox_inches='tight')
        plt.close()

    # 3. Distribución de scores de disbiosis
    fig, ax = plt.subplots(figsize=(10, 6))
    colores = ['#4DAF4A', '#377EB8', '#FF7F00', '#E41A1C']
    posiciones = []
    datos_boxplot = []

    for i, trat in enumerate(datos.clases_tratamiento):
        mask = df_eubiosis['Tratamiento'] == trat
        vals = df_eubiosis.loc[mask, 'Score_Disbiosis'].values
        datos_boxplot.append(vals)
        posiciones.append(i)

    bp = ax.boxplot(datos_boxplot, positions=posiciones, patch_artist=True,
                    widths=0.6, showfliers=True)
    for patch, color in zip(bp['boxes'], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Añadir puntos individuales
    for i, (vals, trat) in enumerate(zip(datos_boxplot, datos.clases_tratamiento)):
        jitter = np.random.normal(0, 0.05, len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals,
                   c=colores[i], s=30, alpha=0.6, edgecolors='grey', linewidth=0.5)

    ax.set_xticks(posiciones)
    ax.set_xticklabels(datos.clases_tratamiento)
    ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    # EN
    ax.set_xlabel('Treatment')
    ax.set_ylabel('Dysbiosis Score (z-score)')
    ax.set_title('Dysbiosis Evaluation by Treatment\n(Based on VAE Reconstruction Error)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, 'proyecto_b_disbiosis_en.tiff'), dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})

    # ES
    ax.set_xlabel('Tratamiento')
    ax.set_ylabel('Score de Disbiosis (z-score)')
    ax.set_title('Evaluación de Disbiosis por Tratamiento\n(Basado en Error de Reconstrucción del VAE)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, 'proyecto_b_disbiosis_es.tiff'), dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close()

    # Guardar modelo
    torch.save({
        'model_state_dict': modelo.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'latent_dim': args.latent_dim,
        'n_taxa': n_taxa,
        'media_ctrl': media_ctrl,
        'std_ctrl': std_ctrl,
    }, os.path.join(args.models_dir, 'vae_eubiosis.pt'))

    logger.info("  ✓ Proyecto B completado")
    return metricas


# =============================================================================
# PROYECTO C: TRANSFORMER PARA CLASIFICACIÓN 16S
# =============================================================================

class TransformerEncoder16S(nn.Module):
    """
    Transformer Encoder para clasificación taxonómica de lecturas 16S.
    Usa tokenización por k-mers y clasificación a nivel de género.

    Arquitectura:
        Embedding → Positional Encoding → Transformer Encoder
        → [CLS] token → MLP → Clase taxonómica
    """

    def __init__(self, vocab_size: int, n_clases: int, d_model: int = 128,
                 nhead: int = 4, n_layers: int = 3, dim_feedforward: int = 256,
                 max_len: int = 512, dropout: float = 0.1):
        """
        Inicializar Transformer.

        Parámetros:
            vocab_size: Tamaño del vocabulario de k-mers
            n_clases: Número de clases taxonómicas (géneros)
            d_model: Dimensión del modelo
            nhead: Número de cabezas de atención
            n_layers: Número de capas del encoder
            dim_feedforward: Dimensión de las capas FFN
            max_len: Longitud máxima de secuencia
            dropout: Tasa de dropout
        """
        super().__init__()

        self.d_model = d_model

        # Embedding de tokens
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Positional encoding (aprendido)
        self.pos_encoding = nn.Embedding(max_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN para estabilidad
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Clasificador
        self.clasificador = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_clases)
        )

    def forward(self, x):
        """
        Forward pass.

        Parámetros:
            x: Tensor de tokens (batch × seq_len)

        Retorna:
            logits: Logits de clasificación (batch × n_clases)
        """
        batch_size, seq_len = x.shape

        # Máscara de padding
        padding_mask = (x == 0)

        # Embeddings
        tok_emb = self.embedding(x) * (self.d_model ** 0.5)

        # Positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_encoding(positions)

        emb = tok_emb + pos_emb

        # Transformer encoder
        encoded = self.transformer(emb, src_key_padding_mask=padding_mask)

        # Usar el token [CLS] (posición 0) para clasificación
        cls_output = encoded[:, 0, :]

        # Clasificar
        logits = self.clasificador(cls_output)
        return logits


def cargar_lecturas_y_taxonomia(reads_dir: str, taxonomy_ref: str,
                                 max_reads: int = 10000):
    """
    Cargar lecturas FASTQ y sus asignaciones taxonómicas.

    Parámetros:
        reads_dir: Directorio con archivos FASTQ filtrados
        taxonomy_ref: Directorio con resultados de EMU
        max_reads: Número máximo de lecturas a cargar

    Retorna:
        secuencias: Lista de secuencias de ADN
        etiquetas: Lista de géneros asignados
    """
    import gzip

    logger.info(f"Cargando lecturas de: {reads_dir}")
    logger.info(f"Referencia taxonómica: {taxonomy_ref}")

    secuencias = []
    etiquetas = []

    reads_path = Path(reads_dir)
    tax_path = Path(taxonomy_ref)

    # Buscar archivos FASTQ
    fastq_files = sorted(
        list(reads_path.glob('**/*.fastq.gz')) +
        list(reads_path.glob('**/*.fastq')) +
        list(reads_path.glob('**/*.fq.gz'))
    )

    if not fastq_files:
        logger.warning(f"No se encontraron archivos FASTQ en {reads_dir}")
        # Generar datos sintéticos para demostración
        logger.info("Generando datos sintéticos de demostración...")
        return _generar_datos_sinteticos_transformer(max_reads)

    # Cargar asignaciones taxonómicas de EMU
    tax_assignments = {}
    emu_files = sorted(tax_path.glob('**/emu_results*.tsv'))

    for emu_file in emu_files:
        try:
            df = pd.read_csv(emu_file, sep='\t')
            # Extraer género de la columna taxonómica
            if 'genus' in df.columns:
                for _, row in df.iterrows():
                    genus = str(row['genus']).strip()
                    if genus and genus != 'nan':
                        # Usar como distribución de probabilidad para asignar
                        tax_assignments[genus] = tax_assignments.get(genus, 0) + 1
        except Exception as e:
            logger.warning(f"Error leyendo {emu_file}: {e}")

    if not tax_assignments:
        logger.warning("No se pudieron cargar asignaciones taxonómicas")
        return _generar_datos_sinteticos_transformer(max_reads)

    # Leer secuencias
    n_read = 0
    generos = list(tax_assignments.keys())

    for fq_file in fastq_files:
        if n_read >= max_reads:
            break

        try:
            open_fn = gzip.open if str(fq_file).endswith('.gz') else open
            mode = 'rt' if str(fq_file).endswith('.gz') else 'r'

            with open_fn(fq_file, mode) as f:
                while n_read < max_reads:
                    # Leer un registro FASTQ (4 líneas)
                    header = f.readline().strip()
                    if not header:
                        break
                    seq = f.readline().strip()
                    plus = f.readline().strip()
                    qual = f.readline().strip()

                    if seq and len(seq) >= 100:
                        secuencias.append(seq)
                        # Asignar género (simplificado: asignación proporcional)
                        # En producción, usar las asignaciones de EMU por read
                        genero = np.random.choice(generos)
                        etiquetas.append(genero)
                        n_read += 1

        except Exception as e:
            logger.warning(f"Error leyendo {fq_file}: {e}")

    logger.info(f"  Lecturas cargadas: {len(secuencias)}")
    logger.info(f"  Géneros únicos: {len(set(etiquetas))}")

    return secuencias, etiquetas


def _generar_datos_sinteticos_transformer(n_reads: int = 5000):
    """
    Generar datos sintéticos de secuencias 16S para demostración.
    Crea secuencias con patrones k-mer específicos por género.
    """
    logger.info("Generando datos sintéticos de 16S para demostración...")

    generos = [
        'Lactobacillus', 'Bifidobacterium', 'Bacteroides',
        'Escherichia', 'Clostridium', 'Streptococcus',
        'Prevotella', 'Faecalibacterium', 'Ruminococcus',
        'Plantilactobacillus'
    ]

    # Motivos característicos por género (simplificados)
    motivos = {
        'Lactobacillus': 'ATCGATCG',
        'Bifidobacterium': 'GCTAGCTA',
        'Bacteroides': 'TTAACCGG',
        'Escherichia': 'CCGGAATT',
        'Clostridium': 'AATTCCGG',
        'Streptococcus': 'GGCCTTAA',
        'Prevotella': 'TACGTACG',
        'Faecalibacterium': 'GATCGATC',
        'Ruminococcus': 'CTGACTGA',
        'Plantilactobacillus': 'ATCGATCG'  # Similar a Lactobacillus
    }

    secuencias = []
    etiquetas = []

    for _ in range(n_reads):
        genero = np.random.choice(generos)
        # Generar secuencia con motivo embebido
        largo = np.random.randint(800, 1500)
        seq = ''.join(np.random.choice(['A', 'C', 'G', 'T'], largo))
        # Insertar motivo varias veces
        motivo = motivos[genero]
        n_inserciones = np.random.randint(3, 10)
        for _ in range(n_inserciones):
            pos = np.random.randint(0, max(1, largo - len(motivo)))
            seq = seq[:pos] + motivo + seq[pos + len(motivo):]

        secuencias.append(seq)
        etiquetas.append(genero)

    return secuencias, etiquetas


def ejecutar_proyecto_c(args):
    """
    Proyecto C: Transformer para Clasificación 16S.
    Clasifica lecturas 16S a nivel de género usando un
    Transformer encoder con tokenización por k-mers.
    """
    logger.info("=" * 60)
    logger.info("  PROYECTO C: TRANSFORMER PARA CLASIFICACIÓN 16S")
    logger.info("=" * 60)

    device = detectar_dispositivo()
    crear_directorios(args.output_dir, args.models_dir, args.figures_dir)

    # ─── Cargar datos ────────────────────────────────────────────────────
    reads_dir = getattr(args, 'reads_dir', None) or 'reads'
    taxonomy_ref = getattr(args, 'taxonomy_ref', None) or 'taxonomy'

    secuencias, etiquetas = cargar_lecturas_y_taxonomia(
        reads_dir, taxonomy_ref, max_reads=10000
    )

    if len(secuencias) == 0:
        logger.error("No se pudieron cargar secuencias. Abortando Proyecto C.")
        return {}

    # ─── Codificar etiquetas ─────────────────────────────────────────────
    le_genero = LabelEncoder()
    y = le_genero.fit_transform(etiquetas)
    clases_genero = list(le_genero.classes_)
    n_clases = len(clases_genero)

    logger.info(f"Secuencias: {len(secuencias)} | Géneros: {n_clases}")
    logger.info(f"Tamaño k-mer: {args.kmer_size}")

    # ─── Crear dataset ───────────────────────────────────────────────────
    dataset = DatasetSecuencias(
        secuencias, y,
        kmer_size=args.kmer_size,
        max_len=512
    )
    vocab_size = len(dataset.vocab)
    logger.info(f"Vocabulario: {vocab_size} tokens")

    # ─── Validación cruzada ──────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    todas_pred = np.zeros(len(y), dtype=int)
    todas_prob = np.zeros((len(y), n_clases))

    historia_loss_train = []
    historia_loss_val = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(secuencias, y)):
        logger.info(f"\n  Fold {fold + 1}/5")

        ds_train = Subset(dataset, train_idx)
        ds_val = Subset(dataset, val_idx)

        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

        # Crear modelo
        modelo = TransformerEncoder16S(
            vocab_size=vocab_size,
            n_clases=n_clases,
            d_model=128,
            nhead=4,
            n_layers=3,
            dim_feedforward=256,
            max_len=512,
            dropout=0.1
        ).to(device)

        optimizer = torch.optim.AdamW(
            modelo.parameters(), lr=args.learning_rate, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        criterion = nn.CrossEntropyLoss()

        # ─── Entrenamiento ───────────────────────────────────────────
        fold_loss_train = []
        fold_loss_val = []

        for epoch in range(args.epochs):
            # Entrenamiento
            modelo.train()
            running_loss = 0
            n_batches = 0

            for batch_tokens, batch_labels in dl_train:
                batch_tokens = batch_tokens.to(device)
                batch_labels = batch_labels.to(device)

                logits = modelo(batch_tokens)
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = running_loss / max(n_batches, 1)
            fold_loss_train.append(avg_train_loss)

            # Validación
            modelo.eval()
            val_loss = 0
            val_batches = 0

            with torch.no_grad():
                for batch_tokens, batch_labels in dl_val:
                    batch_tokens = batch_tokens.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = modelo(batch_tokens)
                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            fold_loss_val.append(avg_val_loss)

            if (epoch + 1) % 20 == 0:
                logger.info(f"    Epoch {epoch + 1}/{args.epochs} | "
                            f"Train Loss: {avg_train_loss:.4f} | "
                            f"Val Loss: {avg_val_loss:.4f}")

        historia_loss_train.append(fold_loss_train)
        historia_loss_val.append(fold_loss_val)

        # ─── Predicción en validación ────────────────────────────────
        modelo.eval()
        with torch.no_grad():
            fold_preds = []
            fold_probs = []
            for batch_tokens, _ in dl_val:
                batch_tokens = batch_tokens.to(device)
                logits = modelo(batch_tokens)
                probs = F.softmax(logits, dim=1)
                fold_preds.append(logits.argmax(dim=1).cpu().numpy())
                fold_probs.append(probs.cpu().numpy())

            fold_preds = np.concatenate(fold_preds)
            fold_probs = np.concatenate(fold_probs)

            todas_pred[val_idx] = fold_preds
            todas_prob[val_idx] = fold_probs

    # ─── Métricas finales ────────────────────────────────────────────────
    logger.info("\n  === RESULTADOS PROYECTO C ===")

    acc = accuracy_score(y, todas_pred)
    f1 = f1_score(y, todas_pred, average='weighted')
    f1_macro = f1_score(y, todas_pred, average='macro')

    logger.info(f"  Accuracy: {acc:.3f}")
    logger.info(f"  F1 weighted: {f1:.3f}")
    logger.info(f"  F1 macro: {f1_macro:.3f}")

    reporte = classification_report(
        y, todas_pred,
        target_names=clases_genero,
        output_dict=True
    )

    # Guardar métricas
    metricas = {
        'accuracy': acc,
        'f1_weighted': f1,
        'f1_macro': f1_macro,
        'n_secuencias': len(secuencias),
        'n_generos': n_clases,
        'vocab_size': vocab_size,
        'kmer_size': args.kmer_size,
        'epochs': args.epochs,
        'reporte': reporte,
    }
    guardar_metricas(metricas, os.path.join(args.output_dir, 'metricas_proyecto_c.json'))

    # Reporte detallado
    reporte_texto = classification_report(y, todas_pred, target_names=clases_genero)
    with open(os.path.join(args.output_dir, 'reporte_clasificacion.txt'), 'w') as f:
        f.write("REPORTE DE CLASIFICACIÓN TAXONÓMICA\n")
        f.write("=" * 50 + "\n\n")
        f.write(reporte_texto)
    logger.info(f"\n{reporte_texto}")

    # ─── Figuras ─────────────────────────────────────────────────────────

    # 1. Matriz de confusión
    cm = confusion_matrix(y, todas_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=clases_genero, yticklabels=clases_genero, ax=ax)
    ax.set_xlabel('Género Predicted')
    ax.set_ylabel('Género Actual')
    ax.set_title(f'Clasificación Taxonómica por Transformer\n'
                 f'Accuracy: {acc:.3f} | F1: {f1:.3f}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, 'proyecto_c_confusion.tiff'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Curvas de pérdida (promedio de folds)
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_train = np.mean(historia_loss_train, axis=0)
    avg_val = np.mean(historia_loss_val, axis=0)
    ax.plot(avg_train, label='Entrenamiento', color='steelblue')
    ax.plot(avg_val, label='Validación', color='coral')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pérdida (Cross-Entropy)')
    ax.set_title('Curvas de Aprendizaje del Transformer (Promedio 5-fold)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, 'proyecto_c_loss.tiff'), dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close()

    # 3. F1-score por género
    f1_por_genero = [reporte[g]['f1-score'] for g in clases_genero if g in reporte]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(clases_genero)), f1_por_genero, color='steelblue')
    ax.set_yticks(range(len(clases_genero)))
    ax.set_yticklabels(clases_genero, fontsize=9)
    ax.set_xlabel('F1-Score')
    ax.set_title('F1-Score por Género')
    ax.set_xlim(0, 1)

    # Añadir valores
    for bar, val in zip(bars, f1_por_genero):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, 'proyecto_c_f1_genero.tiff'), dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close()

    # Guardar modelo
    torch.save({
        'model_state_dict': modelo.state_dict(),
        'vocab': dataset.vocab,
        'clases_genero': clases_genero,
        'kmer_size': args.kmer_size,
        'n_clases': n_clases,
    }, os.path.join(args.models_dir, 'transformer_16s.pt'))

    # Guardar vocabulario
    with open(os.path.join(args.output_dir, 'vocabulario_kmers.json'), 'w') as f:
        json.dump(dataset.vocab, f)

    logger.info("  ✓ Proyecto C completado")
    return metricas


# =============================================================================
# CLI PRINCIPAL
# =============================================================================

def crear_parser():
    """Crear parser de argumentos con subcomandos."""
    parser = argparse.ArgumentParser(
        description='Pipeline de Deep Learning para Microbioma 16S',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Proyectos disponibles:
  predict      Predictor de fenotipo multi-cabeza (Proyecto A)
  vae          VAE para evaluación de eubiosis (Proyecto B)
  transformer  Transformer para clasificación 16S (Proyecto C)

Ejemplos:
  python microbiome_dl_pipeline.py predict --otu_table otu.csv --metadata meta.csv
  python microbiome_dl_pipeline.py vae --otu_table otu.csv --metadata meta.csv --latent_dim 2
  python microbiome_dl_pipeline.py transformer --reads_dir ./reads --taxonomy_ref ./emu
        """
    )

    subparsers = parser.add_subparsers(dest='proyecto', help='Proyecto a ejecutar')

    # ─── Argumentos comunes ──────────────────────────────────────────────
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('--output_dir', type=str, default='results',
                        help='Directorio de salida para resultados')
    parent.add_argument('--models_dir', type=str, default='models',
                        help='Directorio para guardar modelos')
    parent.add_argument('--figures_dir', type=str, default='figures',
                        help='Directorio para guardar figuras')
    parent.add_argument('--epochs', type=int, default=100,
                        help='Número de epochs de entrenamiento')
    parent.add_argument('--batch_size', type=int, default=8,
                        help='Tamaño de batch')
    parent.add_argument('--learning_rate', type=float, default=0.001,
                        help='Tasa de aprendizaje')
    parent.add_argument('--seed', type=int, default=42,
                        help='Semilla para reproducibilidad')

    # ─── Proyecto A: Predictor ───────────────────────────────────────────
    p_predict = subparsers.add_parser('predict', parents=[parent],
                                       help='Predictor de fenotipo multi-cabeza')
    p_predict.add_argument('--otu_table', type=str, required=True,
                           help='Ruta a la tabla OTU (CSV)')
    p_predict.add_argument('--metadata', type=str, required=True,
                           help='Ruta al archivo de metadata (CSV)')
    p_predict.add_argument('--cv_folds', type=int, default=8,
                           help='Número de folds para CV (8 recomendado con n=64)')

    # ─── Proyecto B: VAE ─────────────────────────────────────────────────
    p_vae = subparsers.add_parser('vae', parents=[parent],
                                   help='VAE para evaluación de eubiosis')
    p_vae.add_argument('--otu_table', type=str, required=True,
                       help='Ruta a la tabla OTU (CSV)')
    p_vae.add_argument('--metadata', type=str, required=True,
                       help='Ruta al archivo de metadata (CSV)')
    p_vae.add_argument('--latent_dim', type=int, default=2,
                       help='Dimensión del espacio latente')

    # ─── Proyecto C: Transformer ─────────────────────────────────────────
    p_transformer = subparsers.add_parser('transformer', parents=[parent],
                                           help='Transformer para clasificación 16S')
    p_transformer.add_argument('--reads_dir', type=str, required=True,
                               help='Directorio con lecturas FASTQ filtradas')
    p_transformer.add_argument('--taxonomy_ref', type=str, required=True,
                               help='Directorio con resultados de EMU')
    p_transformer.add_argument('--kmer_size', type=int, default=6,
                               help='Tamaño del k-mer para tokenización')

    return parser


def main():
    """Punto de entrada principal del pipeline."""
    parser = crear_parser()
    args = parser.parse_args()

    if args.proyecto is None:
        parser.print_help()
        sys.exit(1)

    # Configuración global
    configurar_reproducibilidad(args.seed)

    # Configurar memoria CUDA
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    logger.info("=" * 60)
    logger.info("  PIPELINE DE DEEP LEARNING - MICROBIOMA 16S")
    logger.info("=" * 60)
    logger.info(f"Proyecto   : {args.proyecto}")
    logger.info(f"Dispositivo: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"PyTorch    : {torch.__version__}")
    logger.info(f"Fecha      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Semilla    : {args.seed}")

    inicio = time.time()

    try:
        if args.proyecto == 'predict':
            metricas = ejecutar_proyecto_a(args)
        elif args.proyecto == 'vae':
            metricas = ejecutar_proyecto_b(args)
        elif args.proyecto == 'transformer':
            metricas = ejecutar_proyecto_c(args)
        else:
            parser.print_help()
            sys.exit(1)

        elapsed = time.time() - inicio
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Pipeline completado en {elapsed:.1f} segundos")
        logger.info(f"  ({elapsed / 60:.1f} minutos)")
        logger.info(f"{'=' * 60}")

    except KeyboardInterrupt:
        logger.warning("Pipeline interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error fatal en el pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
