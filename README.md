# hybrid-nids-gconvtrans

> Hybrid Network Intrusion Detection System combining a **Graph Convolutional Transformer (GConvTrans)** with a **Stacked Ensemble** (RF · HGB · XGBoost) trained on the **CIC-IDS 2018** dataset.

---

## 📁 Notebooks

| File | Split | Description |
|------|-------|-------------|
| `NIDS_CIC_IDS2018_70_20_10_Split.ipynb` | **70 / 20 / 10** | Primary notebook — train / val / test |
| `NIDS_CIC_IDS2018_65_20_15_Split.ipynb` | **65 / 20 / 15** | Baseline notebook — original split ratio |

Both notebooks share the same architecture and pipeline. The only difference is the train/test allocation ratio, which affects how much data the model sees during training vs. how much is held out for final evaluation.

---

## Overview

Traditional NIDS rely on shallow signature-based rules that fail against novel attacks. This project takes a hybrid deep learning + ensemble approach:

1. **GConvTrans** — builds a k-NN graph over network flow features and applies graph convolution followed by a Transformer encoder, capturing both local neighbourhood structure and global sequence dependencies.
2. **Stacked Ensemble** — trains five base learners (DT, RF, HistGBM, KNN, XGBoost), selects the top 3 by validation Macro-F1, and combines them via a Logistic Regression meta-learner using out-of-fold predictions.
3. A **winner declaration** compares both models across Accuracy, Macro-F1, Weighted-F1, and MCC on the held-out test set.

---

## Dataset

**CIC-IDS 2018** — Canadian Institute for Cybersecurity Intrusion Detection dataset 2018.

| Property | Value |
|----------|-------|
| Source | University of New Brunswick (UNB) |
| Traffic classes | Benign, Bot, Brute Force (Web/XSS), DDoS (HOIC/LOIC-UDP/LOIC-HTTP), DoS (GoldenEye/SlowHTTPTest), FTP-BruteForce, SQL Injection |
| Total classes | 11 |
| Features | 83 network flow features (CICFlowMeter) |
| Format | CSV per day (10 files, ~7 GB total) |

Download from:
- [Kaggle — solarmainframe/ids-intrusion-csv](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv) *(requires account + rule acceptance)*
- [Hugging Face mirror — dnth/cicids2018](https://huggingface.co/datasets/dnth/cicids2018) *(no account needed)*
- [UNB official](https://www.unb.ca/cic/datasets/ids-2018.html)

---

## Pipeline

```
Raw CSVs (10 files, ~7 GB)
        │
        ▼
Chunked Streaming Loader          ← early-exit per file, 10k rows/chunk
        │  50,000 rows sampled
        ▼
Numeric Coercion + Inf/NaN clean  ← CIC-IDS 2018 stores numbers as strings
        │
        ▼
Mutual Information Feature Selection  ← top-12 of 68 candidate features
        │
        ▼
Stratified Split (70/20/10 or 65/20/15)
        │
        ▼
Median Imputation → StandardScaler → SMOTE (train only)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  GConvTrans                      Stacked Ensemble
  k-NN graph → GraphConv          DT · RF · HGB · KNN · XGB
  → Transformer encoder           → OOF meta-features
  → NLL loss · CosineAnnealingLR  → Logistic Regression meta-learner
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
             Winner Declaration
        (Test Macro-F1 as primary metric)
```

---

## 📊 Outputs

### Plots (saved to `Google Drive/NIDS/plots/`)

| File | Description |
|------|-------------|
| `eda_class_dist.png` | Class distribution across all 11 traffic types |
| `eda_feature_dist.png` | Top-12 MI-selected feature histograms |
| `eda_correlation.png` | Pearson correlation heatmap (winsorised p1–p99) |
| `eda_split.png` | Train / Val / Test split pie chart |
| `smote_effect.png` | Class counts before vs after SMOTE |
| `graph_visualisation.png` | k-NN graph structure sample |
| `gconvtrans_training.png` | 4-panel: Loss · Accuracy · F1 · Best-epoch snapshot |
| `all_models_test_performance.png` | All 7 models — Test Accuracy + Macro-F1 |
| `head_to_head_gconv_vs_ensemble.png` | GConvTrans vs Ensemble across Train/Val/Test |
| `winner_declaration.png` | 4-metric comparison with winner highlighted |
| `confusion_matrices.png` | Normalised confusion matrices for both final models |
| `feature_importances.png` | Top-20 RF feature importances |

### Models (saved to `Google Drive/NIDS/models/`)

| File | Description |
|------|-------------|
| `gconvtrans_best.pt` | Best GConvTrans checkpoint (PyTorch state dict) |
| `DT_model.pkl` | Decision Tree |
| `RF_model.pkl` | Random Forest |
| `HGB_model.pkl` | HistGradientBoosting |
| `KNN_model.pkl` | K-Nearest Neighbours |
| `XGB_model.pkl` | XGBoost |
| `meta_learner.pkl` | Stacking meta-learner (Logistic Regression) |
| `scaler.pkl` | StandardScaler (fit on train only) |
| `label_encoder.pkl` | LabelEncoder for class index ↔ name mapping |

---

## How to Run

1. **Open in Google Colab** with a GPU runtime (T4 recommended)
2. **Mount Google Drive** — Cell 3 does this automatically
3. **Download the dataset** — Cell 4 tries Kaggle → HuggingFace → manual upload in order
4. **Run cells in order** (Cell 1 → Cell 13)

>  For the Kaggle download to work, visit the dataset page and accept the rules before running Cell 4, then add your Kaggle API credentials to Colab secrets.

---

## Key Hyperparameters

| Parameter | Value | Location |
|-----------|-------|----------|
| `SAMPLE_SIZE` | 50,000 rows | Cell 5 |
| `CHUNK_SIZE` | 10,000 rows | Cell 5 |
| `SEED` | 42 | Cell 2 |
| GConvTrans hidden dim | 128 | Cell 8 |
| GConvTrans epochs | 150 | Cell 8 |
| GConvTrans patience | 30 | Cell 8 |
| k-NN graph neighbours | 10 | Cell 7 |
| SMOTE k_neighbors | min(5, class_min−1) | Cell 6 |
| OOF folds | 3 | Cell 10 |
| Meta-learner C | 5 | Cell 10 |

---

## Dependencies

```
torch · torch_geometric · sklearn · xgboost
imbalanced-learn · pandas · numpy · matplotlib
seaborn · kagglehub · psutil · joblib
```

All installed in Cell 1 via `pip`.

---

## 📈 Evaluation Metrics

The primary metric for model selection is **Test Macro-F1** — it weights every attack class equally, regardless of how many samples it has, making it more honest than accuracy on imbalanced NIDS data where Benign traffic dominates.

Secondary metrics reported: Weighted-F1, Matthews Correlation Coefficient (MCC), and per-class precision/recall in the confusion matrix.

---

## 👤 Author: Yeng Sebastian

Built for academic research in network security and hybrid deep learning systems.  
Dataset credit: Canadian Institute for Cybersecurity, University of New Brunswick.
