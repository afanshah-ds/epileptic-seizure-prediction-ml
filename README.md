# 🧠 Epileptic Seizure Prediction — Complete ML Pipeline

## 📌 Project Overview
[cite_start]This repository contains a comprehensive Machine Learning pipeline developed as a major semester assignment[cite: 199]. [cite_start]The project systematically evaluates how preprocessing decisions, feature engineering, model complexity, and regularization strategies affect the generalization performance of Logistic Regression models in safety-critical clinical tasks[cite: 195].

### 🔬 Core Research Questions Investigated:
1. [cite_start]Does the sequential order of preprocessing transformations significantly affect metrics? [cite: 13, 253]
2. [cite_start]Which regularization technique ($L_1$, $L_2$, or ElasticNet) generalizes best across varied data structures? [cite: 22, 288]
3. [cite_start]Does ElasticNet consistently outperform pure Lasso or Ridge variants? [cite: 177]
4. [cite_start]How do class-imbalance strategies interact with model regularization? [cite: 180, 347]

---

## 📊 Dataset Framework
[cite_start]Using the benchmark **Bonn University / UCI Epileptic Seizure Recognition dataset**, three distinct sub-datasets were engineered to simulate real-world clinical hurdles[cite: 239, 241]:

| Dataset Configuration | Samples | Features | Class Split | Target Characteristic |
| :--- | :--- | :--- | :--- | :--- |
| **UCI-Epileptic (Real)** | 11,500 | 178 | 80% / 20% | [cite_start]Raw EEG Time-Series Data [cite: 243] |
| **CHB-MIT Proxy (Real)** | 4,600 | 178 | 95% / 5% | [cite_start]Severe Imbalance Clinical Simulation [cite: 243, 247] |
| **Kaggle-Frequency (Real)** | 4,600 | 32 | 50% / 50% | [cite_start]Feature-Engineered (FFT Power Bands) [cite: 243] |

---

## 🛠️ Preprocessing Architecture
[cite_start]Two contrasting pipelines were built to test non-commutative transformations[cite: 51]:
* [cite_start]**Pipeline A (Filter-First):** `StandardScaler` ➔ `4th-order Butterworth Bandpass Filter (0.5–40 Hz)` ➔ `SelectKBest (ANOVA F-test)` [cite: 54, 57, 58, 254]
* [cite_start]**Pipeline B (Extract-then-Reduce):** `Statistical Feature Extraction` ➔ `RobustScaler` ➔ `PCA (15 Components)` [cite: 65, 67, 69, 70, 259]

---

## 📈 Experimental Results Dashboard

### 1. Baseline Model Performance
[cite_start]*Evaluated on Pipeline A processed data ($C=1.0$, $L_2$ penalty, `class_weight='balanced'`):* [cite: 84, 302]

| Dataset | Accuracy | F1-Score | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **UCI-Epileptic** | ~0.88 | ~0.76 | ~0.72 | ~0.92 [cite: 305] |
| **CHB-MIT Proxy** | ~0.95 | ~0.61 | ~0.58 | ~0.89 [cite: 305] |
| **Kaggle-Frequency** | ~0.91 | ~0.89 | ~0.91 | ~0.97 [cite: 305] |

### 2. Regularization Strategy Comparison
*Averaged metrics across all three sub-datasets at optimal configuration strengths:* [cite: 312]

| Penalty Type | Avg F1-Score | Avg PR-AUC | Coefficient Sparsity | Operational Stability |
| :--- | :---: | :---: | :---: | :--- |
| **$L_1$ (Lasso)** | ~0.72 | ~0.69 | High (40% - 70%) | [cite_start]Moderate [cite: 314] |
| **$L_2$ (Ridge)** | ~0.74 | ~0.71 | None (0%) | [cite_start]High [cite: 314] |
| **ElasticNet** | ~0.75 | ~0.72 | Moderate (20% - 50%) | [cite_start]High [cite: 317] |

### 3. Class Imbalance Framework Impact
[cite_start]*Averaged results across all configurations:* [cite: 323]

| Strategy | Avg F1-Score | Precision | Recall | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **No Resampling** | ~0.51 | ~0.79 | ~0.38 | [cite_start]~0.62 [cite: 325] |
| **SMOTE** | ~0.71 | ~0.68 | ~0.75 | [cite_start]~0.74 [cite: 325] |
| **Undersampling** | ~0.68 | ~0.65 | ~0.72 | [cite_start]~0.70 [cite: 325] |
| **Class Weighting** | ~0.73 | ~0.70 | ~0.77 | [cite_start]~0.75 [cite: 325] |

---

## 🏆 Key Research Findings & Takeaways
* [cite_start]**Ordering Matters:** Preprocessing step order is non-commutative[cite: 51, 265]. [cite_start]Pipeline A's filter-first approach handles raw sequential signals beautifully, winning by up to `0.08` in PR-AUC on raw sequences[cite: 78, 330]. [cite_start]Conversely, Pipeline B's extraction strategy performs brilliantly on outlier-heavy imbalanced arrays due to `RobustScaler` handling amplitude shocks[cite: 81, 331].
* [cite_start]**Regularization Champion:** **ElasticNet** consistently serves as the safest default paradigm under dataset parameter uncertainty [cite: 179, 346][cite_start], finding the optimal compromise between $L_1$ column selection sparsity and $L_2$ numerical stability[cite: 153, 357].
* [cite_start]**The Imbalance Bottleneck:** Severe target disparity ($95/5$) completely eclipses basic regularization scaling parameters[cite: 191, 192]. [cite_start]Algorithmic weight tuning or oversampling adaptations must be implemented first before hyperparameter sweeps can work effectively[cite: 183, 351].

---

## 📁 Repository Directory Structure
* `Semester Project Code.ipynb` — Executable end-to-end Python ML notebook pipeline.
* `Semester Project Report(IEEE Format).pdf` — Formally formatted final academic publication paper tracking statistical methodologies.
* `Semester Project Presentation.pptx` — Project review presentation deck including metric evaluation dashboards.
