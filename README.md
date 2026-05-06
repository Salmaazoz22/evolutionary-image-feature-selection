# 🧬 Evolutionary Image Feature Selection for Brain Tumor Classification

> Combining **Genetic Algorithms** and **Particle Swarm Optimization** with **SVM** to intelligently select the most discriminative MRI features — achieving competitive accuracy with dramatically fewer dimensions.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

</div>

---

## 🧠 Overview

Brain tumor classification from MRI scans is a high-stakes medical imaging problem where both **accuracy** and **interpretability** matter. Raw MRI images yield thousands of features when processed with standard descriptors — making direct classification noisy, computationally expensive, and prone to overfitting.

This project tackles that challenge by applying **bio-inspired evolutionary search** to find the minimal, most informative feature subsets from MRI images before training a Support Vector Machine classifier. Two metaheuristic strategies are compared head-to-head:

- **Genetic Algorithm (GA)** — population-based evolutionary search with crossover, mutation, and selection operators
- **Particle Swarm Optimization (PSO)** — swarm-based continuous optimization adapted for binary feature selection

Both approaches are evaluated against a strong PCA+SVM baseline, offering a rigorous empirical study of evolutionary feature selection in a real-world medical imaging context.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🔬 **Feature Extraction** | HOG (8100-dim) + LBP (10-dim) descriptors from 128×128 MRI images |
| 📉 **Dimensionality Reduction** | Full PCA pipeline with 80% / 90% / 95% / 99% variance thresholds |
| 🧬 **Genetic Algorithm** | Tournament/roulette selection, single/two-point/uniform crossover, bitflip/swap mutation, elitism, fitness sharing, early stopping |
| 🐝 **Particle Swarm Optimization** | Binary PSO with sigmoid & V-shaped transfer functions, gbest/lbest topologies, linear/random inertia decay, stagnation-based re-init |
| 🤖 **SVM Classification** | RBF-kernel SVM baseline (~83% accuracy) + fast LinearSVC fitness evaluator inside evolutionary loops |
| 📊 **Streamlit Dashboard** | Interactive UI to configure and run GA/PSO experiments with real-time Plotly visualizations |
| 📈 **Analysis & Visualization** | Publication-ready comparative analysis notebook across all experimental configurations |

---

## 🏗️ Project Structure

```
evolutionary-image-feature-selection/
│
├── notebooks/                         # Sequential ML pipeline notebooks
│   ├── 01_Data_Preprocessing.ipynb    # MRI loading, grayscale, resize, normalization
│   ├── 02_feature_extraction_HOG_LBP.ipynb  # HOG + LBP extraction & StandardScaler
│   ├── 03_pca_transform.ipynb         # Full PCA + variance threshold slicing
│   ├── 04_SVM_Classifier.ipynb        # RBF-SVM baseline + CV evaluation
│   ├── 05_GA_Feature_Selection.ipynb  # GA experiments across 4 configurations (EXP A–D)
│   ├── 06_PSO_Feature_Selection.ipynb # PSO experiments across 4 configurations (EXP 1–4)
│   └── 07_visuals&analysis.ipynb      # Cross-algorithm comparative analysis & plots
│
├── ui/                                # Streamlit interactive dashboard
│   ├── app.py                         # Main Streamlit app (GA + PSO + SVM tabs)
│   ├── ga.py                          # GA engine (configurable operators, fitness, repair)
│   ├── pso.py                         # PSO engine (binary velocity, topologies, inertia)
│   └── svm_eval.py                    # SVM evaluator with feature masking support
│
├── GA_outputs/                        # Saved GA experiment results
│   ├── All Ga EXP/                    # Aggregated: final_results.csv, ga_all_results.csv
│   ├── EXP_A/ … EXP_D/               # Per-experiment: progress & summary CSVs
│
├── PSO_outputs/                       # Saved PSO experiment results
│   ├── Final Results/                 # pso_all_results.csv, pso_summary.csv
│   └── EXP_1/ … EXP_4/               # Per-experiment: progress & summary CSVs
│
├── SVM_Model/
│   └── svm_results_summary.csv        # Baseline SVM metrics
│
├── app.py                             # Root-level Streamlit entry point
└── .gitignore                         # Excludes data/, *.npy, *.pkl, *.joblib
```

> **Note:** Large binary artifacts (`data/`, `*.npy`, `*.pkl`, trained models) are excluded from version control via `.gitignore`. Only code, notebooks, and CSV experiment summaries are tracked.

---

## ⚙️ Pipeline — How It Works

### Step 1 — Data Preprocessing
`01_Data_Preprocessing.ipynb`

- Dataset: [Brain Cancer MRI Dataset](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset) (Kaggle) with **3 classes**: `brain_glioma`, `brain_menin`, `brain_tumor`
- MRI images converted to **grayscale**, resized to **128×128** (Lanczos), normalized to **[0, 1]**
- Optional enhanced pipeline with **CLAHE** contrast enhancement
- Stratified split: **4,844 train** / **1,212 test** samples (class balance preserved: ~1,600 per class)
- Output: `processed_train.npy` — shape `(4844, 128, 128, 1)`

### Step 2 — Feature Extraction
`02_feature_extraction_HOG_LBP.ipynb`

- **HOG** (Histogram of Oriented Gradients): `orientations=9`, `pixels_per_cell=(8,8)`, `cells_per_block=(2,2)`, `block_norm='L2-Hys'` → **8,100 features**
- **LBP** (Local Binary Patterns): `radius=1`, `n_points=8`, `method='uniform'` → **10 features** per histogram
- Concatenation: **8,100 + 10 = 8,110 features** per image
- `StandardScaler` applied across all samples
- Output: `X_features.npy` — shape `(4844, 8110)`

### Step 3 — Dimensionality Reduction (PCA)
`03_pca_transform.ipynb`

- Full PCA (`svd_solver='full'`) on the 8,110-dimensional feature space
- Variance-preserving slices saved at multiple thresholds:

| Threshold | Components Kept |
|---|---|
| 80% variance | **700** components |
| 90% variance | 1,208 components |
| 95% variance | 1,685 components |
| 99% variance | 2,803 components |

- Primary working space: `XPCA_features_80.npy` — shape `(4844, 700)` (80% variance retained)

### Step 4 — SVM Baseline
`04_SVM_Classifier.ipynb`

- Classifier: `SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)`
- 80/20 stratified split (3,875 train / 969 test)
- **Test Accuracy: 83.08%** | **Weighted F1: 82.93%** | **5-fold CV: 83.15% ± 0.70%**

### Step 5 — Evolutionary Feature Selection
`05_GA_Feature_Selection.ipynb` · `06_PSO_Feature_Selection.ipynb`

The evolutionary search is run on the **700 PCA-compressed features**. Each candidate solution is a **binary mask** over those features. Fitness is evaluated with:

```
fitness = α × CV_accuracy + (1 − α) × (1 − n_selected / n_total)
```

where **α = 0.9** balances classification performance against feature sparsity. A fast **LinearSVC with 2-fold CV** (train split only, no test leakage) is used inside the fitness function for speed.

**Genetic Algorithm experiments (EXP A–D):** vary crossover type (single/two-point/uniform), mutation type (bitflip/swap), selection strategy (tournament/roulette), and survivor strategy (elitist/generational).

**PSO experiments (EXP 1–4):** vary transfer function (sigmoid/V-shaped), inertia strategy (fixed/linear decay/random), and topology (gbest/lbest ring).

### Step 6 — Analysis & Reporting
`07_visuals&analysis.ipynb`

Loads all saved CSV artifacts from GA/PSO experiments and generates:
- Convergence curves per experiment
- Accuracy vs. number-of-features trade-off plots
- Cross-algorithm comparison (GA vs PSO vs SVM baseline)
- Per-experiment runtime and stability analysis

---

## 📊 Results & Insights

### Baseline SVM (700 PCA features)

| Metric | Value |
|---|---|
| Test Accuracy | **83.08%** |
| Weighted F1-Score | **82.93%** |
| 5-Fold CV Mean | **83.15%** |
| 5-Fold CV Std | **±0.70%** |

### GA Feature Selection (Best Configurations)

| Metric | Value |
|---|---|
| Mean CV Accuracy | ~72–73% (during search) |
| Final Test Accuracy | ~80–82% |
| Mean Features Selected | ~209–239 out of 700 |
| Feature Reduction | **~66–70%** |
| Mean Runtime | ~166–217 seconds per run |

> Best configuration: **Uniform crossover + Bitflip mutation + Tournament selection + Elitism**

### PSO Feature Selection (Best Configurations)

| Metric | Value |
|---|---|
| Mean CV Accuracy | ~71–72% (during search) |
| Features Selected | ~305 (sigmoid) |
| Feature Reduction | **~56%** |
| Mean Runtime | ~222–632 seconds per run |

> Best configuration: **Sigmoid transfer + Linear inertia decay (gbest topology)**

### Key Takeaways

- **GA achieves greater sparsity** (~66–70% feature reduction) with competitive accuracy, making it preferable when interpretability and computation cost at inference time matter.
- **PSO converges more smoothly** but selects slightly more features; the sigmoid transfer function with linear decay inertia is the most stable configuration.
- Both methods maintain test accuracy within **1–3 percentage points** of the 700-feature baseline while using **less than a third of the features** — a compelling trade-off for clinical deployment scenarios.
- The fitness function design (α=0.9 weighting) effectively balances accuracy and sparsity, preventing degenerate solutions that maximize one at the expense of the other.

---

## 🧰 Tech Stack

| Category | Libraries |
|---|---|
| **Core ML** | `scikit-learn` (SVM, PCA, cross-validation, metrics) |
| **Image Processing** | `opencv-python`, `scikit-image` (HOG, LBP) |
| **Numerical Computing** | `numpy`, `scipy` |
| **Data Handling** | `pandas` |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` |
| **Dashboard** | `streamlit` |
| **Persistence** | `joblib` |
| **Notebooks** | `jupyter` |
| **Language** | Python 3.8+ |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/evolutionary-image-feature-selection.git
cd evolutionary-image-feature-selection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas scikit-learn scikit-image opencv-python \
            matplotlib seaborn plotly streamlit joblib jupyter
```

### 4. Download the dataset

Download the [Brain Cancer MRI Dataset](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset) from Kaggle and place it under `data/`:

```
data/
└── brain-cancer-mri-dataset/
    ├── brain_glioma/
    ├── brain_menin/
    └── brain_tumor/
```

---

## ▶️ Usage

### Run the full pipeline (notebooks in order)

```bash
jupyter notebook
```

Open and execute notebooks sequentially:

```
01_Data_Preprocessing.ipynb       → generates processed_train.npy
02_feature_extraction_HOG_LBP.ipynb → generates X_features.npy
03_pca_transform.ipynb             → generates XPCA_features_80.npy (+ others)
04_SVM_Classifier.ipynb            → baseline SVM evaluation
05_GA_Feature_Selection.ipynb      → GA experiments
06_PSO_Feature_Selection.ipynb     → PSO experiments
07_visuals&analysis.ipynb          → comparative analysis & plots
```

### Launch the interactive dashboard

Place `XPCA_features_80.npy` and `y_labels.npy` in the `ui/` folder, then:

```bash
cd ui
streamlit run app.py
```

The dashboard lets you:
- Configure GA and PSO hyperparameters via sliders/dropdowns
- Run experiments interactively and observe real-time convergence
- Compare GA vs PSO performance head-to-head
- Visualize selected feature masks and accuracy trade-offs

---

## 🔮 Future Work

- **Deep feature extraction** — replace HOG/LBP with CNN embeddings (ResNet, EfficientNet) to capture richer semantic features from MRI scans
- **Multi-objective optimization** — use NSGA-II or MOPSO to explicitly Pareto-optimize the accuracy/sparsity trade-off rather than a weighted scalar
- **Larger search space** — apply evolutionary selection directly on the raw 8,110-feature space or on features extracted from full-resolution images
- **Hyperparameter co-evolution** — simultaneously evolve SVM hyperparameters (C, γ) alongside the feature mask
- **Transfer learning** — fine-tune a pre-trained vision model on the MRI dataset to generate transfer features, then apply GA/PSO selection on those
- **Clinical validation** — extend the dataset, include multi-center MRI studies, and evaluate robustness across acquisition protocols
- **Explainability** — map selected PCA components back to original HOG/LBP spatial regions to visualize which image areas drive classification
- **REST API / Docker deployment** — wrap the trained model and best feature mask into a FastAPI service containerized with Docker

---

## 📁 Data & Artifact Versioning

Large binary files are excluded from Git. After running the full pipeline, your local workspace will contain:

```
data/          ← raw MRI images (downloaded from Kaggle)
*.npy          ← processed features and PCA projections
*.pkl          ← serialized best GA/PSO individuals
models/        ← trained SVM model (joblib)
figures/       ← plots generated by notebooks
```

Consider using [DVC](https://dvc.org/) to version these artifacts alongside the code.



## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

*Built with curiosity, caffeine, and a healthy appreciation for the no-free-lunch theorem.*

</div>
