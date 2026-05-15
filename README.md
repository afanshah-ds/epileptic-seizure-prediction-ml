# 🧠 Epileptic Seizure Prediction — Complete ML Pipeline

## 📌 Project Overview
This repository contains a comprehensive Machine Learning pipeline developed as a major semester assignment. The project systematically evaluates how preprocessing decisions, feature engineering, model complexity, and regularization strategies affect the generalization performance of Logistic Regression models in safety-critical clinical tasks.

### 🔬 Core Research Questions Investigated:
1. Does the sequential order of preprocessing transformations significantly affect metrics?
2. Which regularization technique ($L_1$, $L_2$, or ElasticNet) generalizes best across varied data structures?
3. Does ElasticNet consistently outperform pure Lasso or Ridge variants?
4. How do class-imbalance strategies interact with model regularization?

---

## 📊 Dataset Framework
Using the benchmark **Bonn University / UCI Epileptic Seizure Recognition dataset**, three distinct sub-datasets were engineered to simulate real-world clinical hurdles:

| Dataset Configuration | Samples | Features | Class Split | Target Characteristic |
| :--- | :--- | :--- | :--- | :--- |
| **UCI-Epileptic (Real)** | 11,500 | 178 | 80% / 20% | Raw EEG Time-Series Data |
| **CHB-MIT Proxy (Real)** | 4,600 | 178 | 95% / 5% | Severe Imbalance Clinical Simulation |
| **Kaggle-Frequency (Real)** | 4,600 | 32 | 50% / 50% | Feature-Engineered (FFT Power Bands) |

---

## 🛠️ Preprocessing Architecture
Two contrasting pipelines were built to test non-commutative transformations:
* **Pipeline A (Filter-First):** `StandardScaler` ➔ `4th-order Butterworth Bandpass Filter (0.5–40 Hz)` ➔ `SelectKBest (ANOVA F-test)`
* **Pipeline B (Extract-then-Reduce):** `Statistical Feature Extraction` ➔ `RobustScaler` ➔ `PCA (15 Components)`

---

## 📈 Experimental Results Dashboard

### 1. Baseline Model Performance
*Evaluated on Pipeline A processed data ($C=1.0$, $L_2$ penalty, `class_weight='balanced'`):*

| Dataset | Accuracy | F1-Score | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **UCI-Epileptic** | ~0.88 | ~0.76 | ~0.72 | ~0.92 |
| **CHB-MIT Proxy** | ~0.95 | ~0.61 | ~0.58 | ~0.89 |
| **Kaggle-Frequency** | ~0.91 | ~0.89 | ~0.91 | ~0.97 |

### 2. Regularization Strategy Comparison
*Averaged metrics across all three sub-datasets at optimal configuration strengths:*

| Penalty Type | Avg F1-Score | Avg PR-AUC | Coefficient Sparsity | Operational Stability |
| :--- | :---: | :---: | :---: | :--- |
| **$L_1$ (Lasso)** | ~0.72 | ~0.69 | High (40% - 70%) | Moderate |
| **$L_2$ (Ridge)** | ~0.74 | ~0.71 | None (0%) | High |
| **ElasticNet** | ~0.75 | ~0.72 | Moderate (20% - 50%) | High |

### 3. Class Imbalance Framework Impact
*Averaged results across all configurations:*

| Strategy | Avg F1-Score | Precision | Recall | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **No Resampling** | ~0.51 | ~0.79 | ~0.38 | ~0.62 |
| **SMOTE** | ~0.71 | ~0.68 | ~0.75 | ~0.74 |
| **Undersampling** | ~0.68 | ~0.65 | ~0.72 | ~0.70 |
| **Class Weighting** | ~0.73 | ~0.70 | ~0.77 | ~0.75 |

---

## 🏆 Key Research Findings & Takeaways
* **Ordering Matters:** Preprocessing step order is non-commutative. Pipeline A's filter-first approach handles raw sequential signals beautifully, winning by up to `0.08` in PR-AUC on raw sequences. Conversely, Pipeline B's extraction strategy performs brilliantly on outlier-heavy imbalanced arrays due to `RobustScaler` handling amplitude shocks.
* **Regularization Champion:** **ElasticNet** consistently serves as the safest default paradigm under dataset parameter uncertainty, finding the optimal compromise between $L_1$ column selection sparsity and $L_2$ numerical stability.
* **The Imbalance Bottleneck:** Severe target disparity ($95/5$) completely eclipses basic regularization scaling parameters. Algorithmic weight tuning or oversampling adaptations must be implemented first before hyperparameter sweeps can work effectively.

---

## 📁 Repository Directory Structure
* `Semester Project Code.ipynb` — Executable end-to-end Python ML notebook pipeline.
* `Semester Project Report(IEEE Format).pdf` — Formally formatted final academic publication paper tracking statistical methodologies.
* `Semester Project Presentation.pptx` — Project review presentation deck including metric evaluation dashboards.
