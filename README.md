# Ensemble Machine Learning for Void Filling in Glacier Elevation Change Maps

This repository contains the full codebase and dataset preprocessing pipeline for the study:  
**"An ensemble machine learning approach for filling voids in surface elevation change maps over glacier surfaces"**  
by Cameron Markovsky, Summer Rupper, Simon Brewer, and Richard R. Forster (University of Utah).

## Overview

Glacier mass balance assessments in High Mountain Asia (HMA) require accurate surface elevation change maps, which often contain large voids due to sensor limitations, terrain shadowing, or lack of contrast in accumulation zones. This repository presents:

- A reproducible machine learning-based framework (XGBoost) for filling voids in glacier elevation change maps
- Comparison with traditional constant and hypsometric interpolation methods
- Benchmarking on DEMs from the Eastern and Western Himalayas

## Repository Structure

```
├── data/
│   ├── raw/                  # Raw DEMs and RGI glacier outlines
│   └── processed/            # Filtered and clipped elevation change datasets
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_void_generation.ipynb
│   ├── 03_constant_method.ipynb
│   ├── 04_hypsometric_method.ipynb
│   ├── 05_xgboost_model.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_shap_analysis.ipynb
├── scripts/
│   ├── preprocess_data.py
│   ├── generate_voids.py
│   ├── fill_constant.py
│   ├── fill_hypsometric.py
│   ├── train_xgboost.py
│   ├── evaluate_methods.py
│   └── shap_explain.py
├── models/
│   └── xgboost_model.pkl     # Trained XGBoost model
├── results/
│   ├── figures/
│   └── metrics/
├── environment.yml
└── README.md
```

## Reproducing the Study

### 1. Preprocessing
**Notebook:** `01_preprocessing.ipynb`  
**Script:** `preprocess_data.py`

- Clip Copernicus DEMs and Shean (2020) elevation change maps to RGI v6.0 glacier outlines
- Exclude surge-type and calving glaciers
- Attach glacier-wide attributes: area, slope, aspect, debris cover
- Remove outlier pixels (>3σ from glacier mean)

### 2. Artificial Void Generation
**Notebook:** `02_void_generation.ipynb`  
**Script:** `generate_voids.py`

- Generate voids in upper 50% of glacier elevation distribution
- Radially expand voids from a random seed to simulate realistic accumulation-zone gaps
- Target ~37% glacier-wide void coverage

### 3. Constant Method
**Notebook:** `03_constant_method.ipynb`  
**Script:** `fill_constant.py`

- Fill voids with the **mean surface elevation change** of valid (non-void) pixels per glacier

### 4. Hypsometric Method
**Notebook:** `04_hypsometric_method.ipynb`  
**Script:** `fill_hypsometric.py`

- Bin non-void pixels by elevation (50 m bins)
- Fill voids by assigning mean elevation change in their respective bin
- Use linear interpolation for bins lacking data

### 5. XGBoost Model
**Notebook:** `05_xgboost_model.ipynb`  
**Script:** `train_xgboost.py`

- Use gradient-boosted tree regression (`xgboost`)
- Predict `dh/dt` for void pixels
- Feature inputs:
  - **Per-pixel**: x, y, z
  - **Glacier-wide**: area, slope, sin(aspect), cos(aspect), zmin, zmed, zmax, hypsometric index, debris cover
- Hyperparameter tuning with Optuna (200 trials, early stopping = 50)

### 6. Evaluation
**Notebook:** `06_evaluation.ipynb`  
**Script:** `evaluate_methods.py`

- Evaluate **RMSE** and **MAE** for each method
- Compare void-predicted vs observed values (e.g., KDE plots)
- Assess performance at pixel and glacier-wide levels

### 7. SHAP Analysis
**Notebook:** `07_shap_analysis.ipynb`  
**Script:** `shap_explain.py`

- Use SHAP (SHapley Additive exPlanations) to interpret model predictions
- Visualize spatial influence of predictors on sample glaciers (e.g., Toshain, Langmusang)

## Results Summary

| Region | Method      | RMSE (m a⁻¹) | MAE (m a⁻¹) |
|--------|-------------|--------------|-------------|
| Western | Constant    | 0.357        | 0.271       |
|         | Hypsometric | 0.317        | 0.229       |
|         | XGBoost     | **0.280**    | **0.207**   |
| Eastern | Constant    | 0.510        | 0.392       |
|         | Hypsometric | 0.508        | 0.360       |
|         | XGBoost     | **0.431**    | **0.310**   |

## Study Regions

- **Western Himalaya**: ~8,659 glaciers
- **Eastern Himalaya**: ~2,172 glaciers
- DEM Source: [Shean et al. (2020)](https://doi.org/10.5281/ZENODO.3872696)

## Contact

Cameron Markovsky  
School of Environment, Society, and Sustainability  
University of Utah  
📧 cameron.markovsky@utah.edu
