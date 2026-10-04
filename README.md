# Intrusion Detection & Prevention System (IDPS) - CIC IoT 2023 Pipeline

This repository contains the machine learning and AutoML pipeline for network intrusion detection using the **CIC IoT 2023 Dataset** (Canadian Institute for Cybersecurity, University of New Brunswick). 

The project implements multi-tier classification (binary, 8-class, and fine-grained 34-class attack recognition), Particle Swarm Optimization (PSO) for feature selection, and AutoML model training with deployment artifacts.

---

## 📁 Directory Structure

```plaintext
idps/
├── data/                       # Raw benchmark datasets and percentage distribution samples
├── groupwise_datasets/         # Datasets partitioned by attack category / family
├── pre_processed_dataset/      # Datasets with PSO (Particle Swarm Optimization) selected features
├── notebooks/                  # Interactive Jupyter notebooks for analysis and model training
├── scripts/                    # Standalone Python scripts for ETL, preprocessing, and training
├── models/                     # Trained model artifacts (.pkl, .joblib, feature lists, scalers)
├── logs/                       # Training logs and AutoML leaderboard outputs
├── requirements.txt            # Python dependencies for the environment
└── README.md                   # Project documentation and organization guide
```

---

## 🗺️ Dataset to Notebook & Script Mapping

### 1. Mapping by Notebook & Script (What each file runs on)

| Script / Notebook | Input Dataset(s) | Output Dataset / Artifact | Purpose |
| :--- | :--- | :--- | :--- |
| **`notebooks/MAIN_Preprocessing.ipynb`** | All 10 CSVs in `groupwise_datasets/` | All 10 CSVs in `pre_processed_dataset/` (`*_pso_features.csv`) | Performs data cleaning, handles class imbalance, and executes Particle Swarm Optimization (PSO) feature selection. |
| **`notebooks/MAIN_model_building.ipynb`** | All 10 CSVs in `pre_processed_dataset/` | Model evaluation metrics & comparison | Trains and evaluates ML classifiers (XGBoost, Random Forest, CatBoost, LightGBM) on the reduced PSO feature sets. |
| **`notebooks/MAIN_very_NEW.ipynb`** | All 10 CSVs in `groupwise_datasets/` | `models/Artificats/<dataset>_artifacts/` (models, scalers, encoders, feature lists) | Full end-to-end model training, hyperparameter optimization, and deployment artifact packaging per attack group. |
| **`notebooks/binary_class_optimized.ipynb`** | `data/0.01_dist_percent_2classes.csv` | Binary classification metrics | Hyperparameter optimization and testing for Benign vs. Malicious traffic. |
| **`notebooks/class8.ipynb`** | `data/0.01_dist_percent_8classes.csv` | `engineered_data2.csv` | Exploratory data analysis, 8-class attack classification, and feature engineering. |
| **`scripts/extra.py`** | `data/0.001_dist_percent_34classes.csv` | `data/grouped_attack_labels.csv` | Remaps 34 raw CIC IoT labels into logical attack groups and granular Reconnaissance categories. |
| **`scripts/Extract2.py`** | `data/grouped_attack_labels.csv` | `groupwise_datasets/grouped_attack_labels_new.csv` | Filters out non-relevant attacks to produce a clean multi-class grouped attack dataset. |
| **`scripts/Extract2(1).py`** *(alias: `extract_web_attacks.py`)* | `data/Web_final_dataset.csv` | `groupwise_datasets/Client_Server_Web_Attacks.csv` | Filters browser hijacking and uploading attacks into a dedicated web attack dataset. |
| **`scripts/process_data.py`** | `data/0.01_dist_percent_2classes.csv` | `groupwise_datasets/binary_modified_new.csv` | Cleans binary label format by mapping boolean `True`/`False` to `'benign'`/`'attack'`. |
| **`scripts/binary_class_classification.py`** | `data/0.01_dist_percent_2classes.csv` | Console metrics & confusion matrices | Standalone pipeline with SMOTE oversampling, PSO feature selection, BayesSearchCV, and advanced tree models. |
| **`scripts/test_7_class.py`** | `data/grouped_attack_labels.csv` | `models/classification_artifacts.joblib` | Executes automated IDPS training pipeline via `automlplforidps`. |
| **`scripts/shap_test.py`** | `groupwise_datasets/Service_OS_Discovery.csv`, `models/classification_artifacts.joblib` | SHAP interpretability values | Computes feature importance and explanation values using TreeExplainer. |

---

### 2. Mapping by Dataset (Where each dataset belongs & what uses it)

#### A. Raw & Benchmark Distribution Datasets (`data/`)

| Dataset File | Associated Script / Notebook | Role in Project |
| :--- | :--- | :--- |
| `0.001_dist_percent_34classes.csv` | `scripts/extra.py` | 0.1% sample of all 34 classes; used to generate `grouped_attack_labels.csv`. |
| `0.005_dist_percent_34classes.csv` | Scalability testing / Benchmark | 0.5% sample of all 34 classes. |
| `0.01_dist_percent_2classes.csv` | `notebooks/binary_class_optimized.ipynb`, `scripts/binary_class_classification.py`, `scripts/process_data.py` | 1.0% sample for binary classification (`benign` vs `attack`). |
| `0.01_dist_percent_8classes.csv` | `notebooks/class8.ipynb` | 1.0% sample categorized into 8 core attack classes. |
| `0.01_dist_percent_34classes.csv` | Benchmark testing | 1.0% sample across all 34 fine-grained classes. |
| `0.05_dist_percent_34classes.csv` | Scalability testing | 5.0% sample across all 34 classes (~553.5 MB). |
| `0.1_dist_percent_34classes.csv` | Scalability testing | 10.0% sample across all 34 classes (~1.05 GB). |
| `8class_new_dataset.csv` | Multi-class benchmark | Curated 8-class attack dataset. |
| `grouped_attack_labels.csv` | `scripts/Extract2.py`, `scripts/test_7_class.py` | Produced by `extra.py`; input to AutoML pipeline and group extractor. |
| `grouped_attack_labels_final.csv` | Full-scale training benchmark | Full 1 GB scale version of `grouped_attack_labels.csv`. |
| `Recon_final_dataset.csv` | Legacy / Base dataset | Pre-split dataset for reconnaissance (split into `Host_Discovery` and `Service_OS_Discovery`). |
| `Web_final_dataset.csv` | `scripts/Extract2(1).py` | Pre-split dataset for web attacks (split into `Client_Server_Web_Attacks` and `Injection_Attacks`). |

#### B. Groupwise Attack Family Datasets (`groupwise_datasets/`)
*All 10 datasets in this folder are consumed by:*
* `notebooks/MAIN_Preprocessing.ipynb` (to produce the PSO feature-selected datasets in `pre_processed_dataset/`)
* `notebooks/MAIN_very_NEW.ipynb` (to train models and export artifacts to `models/Artificats/`)

| Dataset File | Generated By / Source | Used By |
| :--- | :--- | :--- |
| `binary_modified_new.csv` | `scripts/process_data.py` | `MAIN_Preprocessing.ipynb`, `MAIN_very_NEW.ipynb` |
| `grouped_attack_labels_new.csv` | `scripts/Extract2.py` | `MAIN_Preprocessing.ipynb`, `MAIN_very_NEW.ipynb` |
| `Client_Server_Web_Attacks.csv` | `scripts/Extract2(1).py` | `MAIN_Preprocessing.ipynb`, `MAIN_very_NEW.ipynb` |
| `DDoS_final_dataset.csv` | Attack family partition | `MAIN_Preprocessing.ipynb`, `MAIN_very_NEW.ipynb` |
| `DoS_final_dataset.csv` | Attack family partition | `MAIN_Preprocessing.ipynb`, `MAIN_very_NEW.ipynb` |
| `Host_Discovery.csv` | Reconnaissance partition | `MAIN_Preprocessing.ipynb`, `MAIN_very_NEW.ipynb` |
| `Injection_Attacks.csv` | Web attack partition | `MAIN_Preprocessing.ipynb`, `MAIN_very_NEW.ipynb` |
| `Mirai_final_dataset.csv` | Botnet partition | `MAIN_Preprocessing.ipynb`, `MAIN_very_NEW.ipynb` |
| `Service_OS_Discovery.csv` | Reconnaissance partition | `MAIN_Preprocessing.ipynb`, `MAIN_very_NEW.ipynb`, `scripts/shap_test.py` |
| `Spoofing_final_dataset.csv` | Attack family partition | `MAIN_Preprocessing.ipynb`, `MAIN_very_NEW.ipynb` |

#### C. PSO Feature-Selected Datasets (`pre_processed_dataset/`)
*All 10 datasets in this folder are generated by `notebooks/MAIN_Preprocessing.ipynb` and consumed by `notebooks/MAIN_model_building.ipynb`.*

| Reduced Dataset File | Generated By | Consumed By | Selected Features |
| :--- | :--- | :--- | :---: |
| `binary_modified_new_pso_features.csv` | `MAIN_Preprocessing.ipynb` | `MAIN_model_building.ipynb` | 26 features |
| `grouped_attack_labels_new_pso_features.csv` | `MAIN_Preprocessing.ipynb` | `MAIN_model_building.ipynb` | 28 features |
| `Client_Server_Web_Attacks_pso_features.csv` | `MAIN_Preprocessing.ipynb` | `MAIN_model_building.ipynb` | 28 features |
| `DDoS_final_dataset_pso_features.csv` | `MAIN_Preprocessing.ipynb` | `MAIN_model_building.ipynb` | 19 features |
| `DoS_final_dataset_pso_features.csv` | `MAIN_Preprocessing.ipynb` | `MAIN_model_building.ipynb` | 17 features |
| `Host_Discovery_pso_features.csv` | `MAIN_Preprocessing.ipynb` | `MAIN_model_building.ipynb` | 20 features |
| `Injection_Attacks_pso_features.csv` | `MAIN_Preprocessing.ipynb` | `MAIN_model_building.ipynb` | 20 features |
| `Mirai_final_dataset_pso_features.csv` | `MAIN_Preprocessing.ipynb` | `MAIN_model_building.ipynb` | 19 features |
| `Service_OS_Discovery_pso_features.csv` | `MAIN_Preprocessing.ipynb` | `MAIN_model_building.ipynb` | 20 features |
| `Spoofing_final_dataset_pso_features.csv` | `MAIN_Preprocessing.ipynb` | `MAIN_model_building.ipynb` | 29 features |

---

## 🔄 End-to-End Data & Model Pipeline

```
                             [Raw CIC IoT 2023 Datasets (data/)]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
     [data/0.001_dist_percent_34classes.csv]         [data/0.01_dist_percent_2classes.csv]
                      │                                               │
               (scripts/extra.py)                           (scripts/process_data.py)
                      │                                               │
                      ▼                                               ▼
        [data/grouped_attack_labels.csv]            [groupwise_datasets/binary_modified_new.csv]
                      │
              (scripts/Extract2.py)
                      │
                      ▼
 [groupwise_datasets/grouped_attack_labels_new.csv]
                      │
                      ├───────────────────────────────────────────────┐
                      ▼                                               ▼
         [groupwise_datasets/*.csv]                      [groupwise_datasets/*.csv]
         (All 10 Category Datasets)                      (All 10 Category Datasets)
                      │                                               │
        (MAIN_Preprocessing.ipynb)                           (MAIN_very_NEW.ipynb)
                      │                                               │
                      ▼                                               ▼
         [pre_processed_dataset/*.csv]                  [models/Artificats/*]
         (10 PSO Feature-Selected CSVs)                 (pkl, scaler, encoder, json)
                      │
         (MAIN_model_building.ipynb)
                      │
                      ▼
          [Model Evaluation Metrics]
```

---

## 📂 Detailed File Catalog

### 1. `data/` — Raw & Benchmark Distribution Datasets
* **Sampled percentage splits:**
  * `0.001_dist_percent_34classes.csv` (0.1% sample, 34 classes)
  * `0.005_dist_percent_34classes.csv` (0.5% sample, 34 classes)
  * `0.01_dist_percent_2classes.csv` (1.0% sample, binary classification)
  * `0.01_dist_percent_8classes.csv` (1.0% sample, 8 classes)
  * `0.01_dist_percent_34classes.csv` (1.0% sample, 34 classes)
  * `0.05_dist_percent_34classes.csv` (5.0% sample, 34 classes)
  * `0.1_dist_percent_34classes.csv` (10.0% sample, 34 classes)
* **Aggregated / Pre-split datasets:**
  * `8class_new_dataset.csv`
  * `grouped_attack_labels.csv` & `grouped_attack_labels_final.csv`
  * `Recon_final_dataset.csv` & `Web_final_dataset.csv`

### 2. `groupwise_datasets/` — Category-Specific Datasets
Specialized datasets separated by attack family/group (48 features each, including `label` and `GroupLabel`).
*(Note: A symlink `groupwise-dataset` points to `groupwise_datasets` for backward compatibility).*

### 3. `pre_processed_dataset/` — PSO Feature-Selected Datasets
Datasets produced by Particle Swarm Optimization (`zoofs.ParticleSwarmOptimization`), reducing 47 features down to optimal predictive subsets (17–29 features).

### 4. `notebooks/` — Jupyter Notebooks
* `MAIN_Preprocessing.ipynb`: Feature selection via Particle Swarm Optimization (PSO).
* `MAIN_model_building.ipynb`: Multi-model training and evaluation on PSO-selected features.
* `MAIN_very_NEW.ipynb`: End-to-end model building, hyperparameter search, and artifact export.
* `binary_class_optimized.ipynb`: Binary classification optimization experiments.
* `class8.ipynb`: 8-class classification exploratory analysis and feature engineering.

### 5. `scripts/` — Python Source Code
* `binary_class_classification.py`: Standalone binary classification pipeline.
* `shap_test.py`: Model interpretability using SHAP values (TreeExplainer).
* `test_7_class.py`: Automated pipeline training and evaluation via `automlplforidps`.
* `process_data.py`: Label conversion utility for boolean benign flags.
* `Extract2.py`: Attack group filter script.
* `Extract2(1).py` *(alias: `extract_web_attacks.py`)*: Web attack extraction script.
* `extra.py`: Granular label remapping script.

### 6. `models/` — Saved Models & Deployment Artifacts
* `Artificats.zip`: The archive containing trained models for all attack groups.
* `Artificats/` (extracted): Subdirectories for each attack family (`model.pkl`, `scaler.pkl`, `target_encoder.pkl`, `feature_list.json`).
* `classification_artifacts.joblib`: Serialized artifacts output by `automlplforidps`.
*(Convenience symlinks `artifacts` and `Artificats` at project root preserve compatibility).*

### 7. `logs/` — Execution Outputs
* `autosklearn-web-output.txt`: Auto-Sklearn optimization log and classification report.

---

## 🚀 Setup & Execution

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Running Notebooks
Launch Jupyter Lab/Notebook from the project root or from `notebooks/`:
```bash
jupyter lab
```
*All relative data paths (`../groupwise_datasets`, `../pre_processed_dataset`, `../data`) are preserved and linked.*
