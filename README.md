# SMM-MOGA: Surrogate-Assisted Multi-Objective Genetic Algorithm

SMM-MOGA searches for neural network (MLP) designs that score well **and** stay small. It uses a true Pareto multi-objective genetic algorithm (NSGA-II style) that balances **macro F1** against **parameter count**. A Random Forest surrogate model and SHAP gene importance steer the mutations.

## What it compares

The benchmark runs four search methods with the same budget (population 20 × 15 generations):

| Method | Idea |
|---|---|
| Random Search | Baseline: sample random architectures |
| Standard GA | Single-objective GA on validation F1 |
| GA + Surrogate | GA with a Random Forest surrogate that predicts fitness |
| **SMM-MOGA** | Pareto selection (non-dominated sorting + crowding distance) with SHAP-guided mutation |

**Scale:** 3 datasets × 4 methods × 5 runs = **60 experiments**. The results are compared with Wilcoxon signed-rank tests.

## Search space

Each chromosome has 6 genes, including the optimizer type, which gives a search space of **1,728** MLP configurations.

## Datasets

- **BRFSS 1998:** a health survey, loaded from `BRFSS1998.csv`
- **Pima Indians Diabetes:** loaded from `diabetes.csv`
- **UCI Heart Disease:** downloaded automatically

## Methodology

- 60 / 20 / 20 train / validation / test split. The test set is used **once**, for the final score.
- Preprocessing without data leakage: imputation, IQR outlier capping and scaling are fitted on the training data only.
- SMOTE oversampling on the training split only.
- An ablation study across 4 variants isolates the effect of each component.

## Outputs

- `ablation_f1_comparison.png`, `ablation_stepup.png`, `ablation_boxplot.png`: ablation results
- `pareto_front.png`: F1 vs. parameter count
- `shap_importance.png`: how gene importance changes over the generations
- `convergence_curve.png`: best F1 per generation
- `results_final.csv`: raw results

## Running

```bash
pip install numpy pandas tensorflow scikit-learn imbalanced-learn scipy shap matplotlib
python smm_moga_benchmark.py
```

The script is built for Kaggle notebooks. It has a keep-alive thread to prevent timeouts and saves a JSON checkpoint after every run, so it resumes automatically after a crash.

## Tech

Python · TensorFlow/Keras · scikit-learn · imbalanced-learn (SMOTE) · SHAP · SciPy · Matplotlib

---
Author: [Mujahed Alabed](https://github.com/Muj-04)
