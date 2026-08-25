# Microbiome Deep Learning & MICOM Pipeline

This repository contains the computational pipeline used to analyse the colonisation dynamics of *L. plantarum* using 16S sequencing, Sparse Autoencoders, and Genome-Scale Metabolic Modelling (MICOM).

## Structure
- `scripts/`: Contains Python, R, and Bash scripts for data preprocessing, statistical analysis, and machine learning.
- `data/`: Place your input data here (e.g., abundance tables, metadata, phylogenetic trees).
- `figures/`: Output directory for the generated high-resolution plots (SVG and TIFF).

## Pipeline Overview
1. **Preprocessing & Diversity Analysis:** Compositional data analysis (CoDA) via CLR transformations and Aitchison distances.
2. **Deep Learning (Predictive Modeling):** Multi-head MLP for phenotypic classification (sex and treatment) and biomarker extraction (Feature Importance).
3. **Latent Space Embedding:** Sparse Autoencoders to discover higher-order microbial functional guilds.
4. **Metabolic Modeling:** MICOM simulations (GLPK optimization) to predict *in silico* growth rates and syntrophic cross-feeding.
5. **Evolutionary Trajectories:** Spectral Embedding and PhILR balances to infer evolutionary pressures and ecosystem maturation.
