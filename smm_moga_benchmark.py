"""
╔══════════════════════════════════════════════════════════════════════╗
║   SMM-MOGA Publication Benchmark                                     ║
║   3 Datasets × 3 Methods × 5 Runs = 45 Experiments                  ║
║                                                                      ║
║   Datasets: BRFSS 1998, UCI Heart Disease, Pima Diabetes             ║
║   Methods:  Random Search, Standard GA, SMM-MOGA (ours)             ║
║   Output:   Results table (mean±std) + Wilcoxon test + plots         ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")
import os

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
GENE_NAMES = ["neurons_layer1", "neurons_layer2", "dropout_rate",
              "learning_rate", "batch_size"]
GENE_CHOICES = [
    [16, 32, 64, 128],
    [16, 32, 64, 128],
    [0.0, 0.1, 0.2, 0.3],
    [0.001, 0.005, 0.01],
    [16, 32, 64],
]

POP_SIZE       = 20
GENERATIONS    = 15
MUTATION_RATE  = 0.4
EPOCHS_GA      = 30
META_MIN_SAMPLES = 20
NUM_RUNS       = 5       # عدد التشغيلات لكل تجربة
METHODS        = ["Random Search", "Standard GA", "SMM-MOGA"]

# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING FUNCTIONS
# ══════════════════════════════════════════════════════════════

def load_brfss(path="BRFSS1998.csv"):
    """BRFSS 1998 — 3-class health prediction, 14 features"""
    print(f"  Loading BRFSS from {path}...")
    df = pd.read_csv(path)

    # Label engineering
    df["risk_score"] = (
        (df["bphigh"] == 1).astype(int) +
        (df["smoke100"] == 1).astype(int) +
        (df["exerany"] == 2).astype(int) +
        (df["hlthplan"] == 2).astype(int)
    )
    df["label"] = df["risk_score"].apply(
        lambda s: "Unhealthy" if s >= 3 else ("At_Risk" if s == 2 else "Healthy")
    )
    df.drop(columns=["risk_score"], inplace=True)

    # Features (no data leakage)
    feature_names = ["age", "_bmi", "checkup", "sex", "drinkany",
                     "educa", "income2", "physhlth", "menthlth",
                     "genhlth", "diabetes", "bloodcho", "seatbelt", "flushot"]

    df[feature_names] = df[feature_names].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=feature_names, inplace=True)

    X = df[feature_names].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    n_classes = len(le.classes_)

    print(f"    Shape: {X.shape}, Classes: {n_classes} {list(le.classes_)}")
    return X, y, n_classes, feature_names, "BRFSS 1998"


def load_diabetes(path="diabetes.csv"):
    """Pima Indians Diabetes — binary, 8 features"""
    print(f"  Loading Diabetes from {path}...")
    df = pd.read_csv(path)

    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    X = df[feature_names].values
    y = df["Outcome"].values
    n_classes = 2

    print(f"    Shape: {X.shape}, Classes: {n_classes}")
    return X, y, n_classes, feature_names, "Pima Diabetes"


def load_heart(path=None):
    """UCI Heart Disease (Cleveland 303) — pulled directly from UCI servers"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

    print(f"  Loading Heart Disease from UCI server...")
    df = pd.read_csv(url, names=columns, na_values='?')
    df.dropna(inplace=True)

    # Binary target: 0 = no disease, 1+ = disease
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                     'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    df[feature_names] = df[feature_names].apply(pd.to_numeric, errors='coerce')

    X = df[feature_names].values
    y = df['target'].values
    n_classes = 2

    print(f"    Shape: {X.shape}, Classes: {n_classes} (0={sum(y==0)}, 1={sum(y==1)})")
    return X, y, n_classes, feature_names, "UCI Heart Disease"


# ══════════════════════════════════════════════════════════════
# 2. CORE ALGORITHM FUNCTIONS
# ══════════════════════════════════════════════════════════════

def chromosome_to_vector(chrom):
    return [float(g) for g in chrom]

def random_chromosome():
    return [random.choice(choices) for choices in GENE_CHOICES]

def build_mlp(chrom, n_features, n_classes):
    neurons1, neurons2, dropout, lr, _ = chrom
    model = Sequential([
        Dense(int(neurons1), activation="relu", input_shape=(n_features,)),
        Dropout(float(dropout)),
        Dense(int(neurons2), activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def evaluate_chromosome(chrom, X_train, y_train_cat, X_test, y_test, n_classes):
    """Train MLP and return (fitness, accuracy, params)"""
    _, _, _, _, batch_size = chrom
    model = build_mlp(chrom, X_train.shape[1], n_classes)
    total_params = model.count_params()

    early_stop = EarlyStopping(monitor='val_loss', patience=3,
                               restore_best_weights=True)
    model.fit(X_train, y_train_cat,
              epochs=EPOCHS_GA, batch_size=int(batch_size),
              verbose=0, validation_split=0.1,
              callbacks=[early_stop])

    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, preds)
    fitness = acc - (total_params / 20000.0) * 0.05

    tf.keras.backend.clear_session()
    return fitness, acc, total_params

def build_meta_model(history):
    X_meta = np.array([h[0] for h in history])
    y_meta = np.array([h[1] for h in history])
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_meta, y_meta)
    return rf

def shap_gene_importance(meta_model, history):
    X_meta = np.array([h[0] for h in history])
    explainer = shap.TreeExplainer(meta_model)
    shap_values = explainer.shap_values(X_meta)
    importance = np.mean(np.abs(shap_values), axis=0)
    return importance

def shap_guided_mutation(chrom, gene_importance):
    child = chrom.copy()
    inv_importance = 1.0 / (gene_importance + 1e-8)
    mutation_probs = inv_importance / inv_importance.sum()
    gene_idx = np.random.choice(len(child), p=mutation_probs)
    child[gene_idx] = random.choice(GENE_CHOICES[gene_idx])
    return child

def random_mutation(chrom):
    child = chrom.copy()
    gene_idx = random.randint(0, len(child) - 1)
    child[gene_idx] = random.choice(GENE_CHOICES[gene_idx])
    return child


# ══════════════════════════════════════════════════════════════
# 3. THREE METHODS
# ══════════════════════════════════════════════════════════════

def run_random_search(X_train, y_train_cat, X_test, y_test, n_classes, seed):
    """Baseline 1: Random Search — evaluate 300 random architectures"""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    total_evals = POP_SIZE * GENERATIONS  # 300
    best_acc = 0
    best_params = 0

    for i in range(total_evals):
        chrom = random_chromosome()
        _, acc, params = evaluate_chromosome(chrom, X_train, y_train_cat,
                                              X_test, y_test, n_classes)
        if acc > best_acc:
            best_acc = acc
            best_params = params

    return best_acc, best_params


def run_standard_ga(X_train, y_train_cat, X_test, y_test, n_classes, seed):
    """Baseline 2: Standard GA with RANDOM mutation (no SHAP)"""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    population = [random_chromosome() for _ in range(POP_SIZE)]
    best_acc = 0
    best_params = 0

    for gen in range(GENERATIONS):
        scores = []
        for chrom in population:
            fit, acc, params = evaluate_chromosome(chrom, X_train, y_train_cat,
                                                    X_test, y_test, n_classes)
            scores.append((chrom, fit, acc, params))

        scores.sort(key=lambda x: x[1], reverse=True)

        if scores[0][2] > best_acc:
            best_acc = scores[0][2]
            best_params = scores[0][3]

        # Selection
        selected = [chrom for chrom, _, _, _ in scores[:POP_SIZE // 2]]

        # Crossover + RANDOM mutation
        new_population = selected.copy()
        while len(new_population) < POP_SIZE:
            p1, p2 = random.sample(selected, 2)
            child = [random.choice([p1[i], p2[i]]) for i in range(len(p1))]
            if random.random() < MUTATION_RATE:
                child = random_mutation(child)
            new_population.append(child)
        population = new_population

    return best_acc, best_params


def run_smm_moga(X_train, y_train_cat, X_test, y_test, n_classes, seed):
    """Our Method: SMM-MOGA with SHAP-guided mutation"""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    population = [random_chromosome() for _ in range(POP_SIZE)]
    history = []
    meta_model = None
    gene_importance = np.ones(len(GENE_NAMES)) / len(GENE_NAMES)

    best_acc = 0
    best_params = 0

    for gen in range(GENERATIONS):
        scores = []
        for chrom in population:
            fit, acc, params = evaluate_chromosome(chrom, X_train, y_train_cat,
                                                    X_test, y_test, n_classes)
            scores.append((chrom, fit, acc, params))
            history.append((chromosome_to_vector(chrom), fit))

        scores.sort(key=lambda x: x[1], reverse=True)

        if scores[0][2] > best_acc:
            best_acc = scores[0][2]
            best_params = scores[0][3]

        # Meta-Model + SHAP
        if len(history) >= META_MIN_SAMPLES:
            meta_model = build_meta_model(history)
            gene_importance = shap_gene_importance(meta_model, history)

        # Selection
        selected = [chrom for chrom, _, _, _ in scores[:POP_SIZE // 2]]

        # Crossover + SHAP-guided mutation
        new_population = selected.copy()
        while len(new_population) < POP_SIZE:
            p1, p2 = random.sample(selected, 2)
            child = [random.choice([p1[i], p2[i]]) for i in range(len(p1))]
            if random.random() < MUTATION_RATE:
                if meta_model is not None:
                    child = shap_guided_mutation(child, gene_importance)
                else:
                    child = random_mutation(child)
            new_population.append(child)
        population = new_population

    return best_acc, best_params


# ══════════════════════════════════════════════════════════════
# 4. MAIN BENCHMARK LOOP
# ══════════════════════════════════════════════════════════════

METHOD_FUNCS = {
    "Random Search": run_random_search,
    "Standard GA":   run_standard_ga,
    "SMM-MOGA":      run_smm_moga,
}

if __name__ == "__main__":

    print("="*70)
    print("  SMM-MOGA PUBLICATION BENCHMARK")
    print(f"  3 Datasets × 3 Methods × {NUM_RUNS} Runs = {3*3*NUM_RUNS} Experiments")
    print("="*70)

    # ── Load all datasets ──
    print("\n📦 Loading datasets...")
    datasets = []

    # تأكد من المسارات الصحيحة لملفاتك
    try:
        datasets.append(load_brfss("BRFSS1998.csv"))
    except FileNotFoundError:
        print("  ⚠️ BRFSS1998.csv not found, skipping...")

    try:
        datasets.append(load_diabetes("diabetes.csv"))
    except FileNotFoundError:
        print("  ⚠️ diabetes.csv not found, skipping...")

    try:
        datasets.append(load_heart())  # يسحب من UCI مباشرة
    except Exception as e:
        print(f"  ⚠️ Heart Disease loading failed: {e}")

    if not datasets:
        print("❌ No datasets found! Check file paths.")
        exit(1)

    # ── Results storage ──
    all_results = {}  # {(dataset_name, method): [acc1, acc2, ...]}
    timing = {}       # {(dataset_name, method): total_seconds}

    # ── Run all experiments ──
    for X, y, n_classes, feature_names, dataset_name in datasets:
        print(f"\n{'═'*70}")
        print(f"  DATASET: {dataset_name}")
        print(f"{'═'*70}")

        for method_name in METHODS:
            print(f"\n  ── Method: {method_name} ──")
            accs = []
            start_time = time.time()

            for run in range(NUM_RUNS):
                seed = 42 + run * 10  # seeds: 42, 52, 62, 72, 82

                # Fresh split each run
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, stratify=y, random_state=seed)
                y_train_cat = to_categorical(y_train, num_classes=n_classes)
                y_test_cat  = to_categorical(y_test,  num_classes=n_classes)

                print(f"    Run {run+1}/{NUM_RUNS} (seed={seed})...", end=" ", flush=True)
                acc, params = METHOD_FUNCS[method_name](
                    X_train, y_train_cat, X_test, y_test, n_classes, seed)
                accs.append(acc)
                print(f"Acc: {acc:.4f} | Params: {params}")

            elapsed = time.time() - start_time
            all_results[(dataset_name, method_name)] = accs
            timing[(dataset_name, method_name)] = elapsed

            mean_acc = np.mean(accs)
            std_acc  = np.std(accs)
            print(f"    ✅ {method_name}: {mean_acc:.4f} ± {std_acc:.4f} ({elapsed:.0f}s)")

    # ══════════════════════════════════════════════════════════════
    # 5. RESULTS TABLE
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  FINAL RESULTS TABLE")
    print("="*70)

    dataset_names = [d[4] for d in datasets]

    # Header
    header = f"{'Method':<18}"
    for dn in dataset_names:
        header += f" | {dn:^22}"
    print(header)
    print("-" * len(header))

    for method in METHODS:
        row = f"{method:<18}"
        for dn in dataset_names:
            accs = all_results.get((dn, method), [])
            if accs:
                row += f" | {np.mean(accs):.4f} ± {np.std(accs):.4f}  "
            else:
                row += f" | {'N/A':^22}"
        print(row)

    # ══════════════════════════════════════════════════════════════
    # 6. STATISTICAL TESTS (Wilcoxon)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  STATISTICAL SIGNIFICANCE (Wilcoxon signed-rank test)")
    print("="*70)

    for dn in dataset_names:
        print(f"\n  Dataset: {dn}")
        smm_accs = all_results.get((dn, "SMM-MOGA"), [])
        for baseline in ["Random Search", "Standard GA"]:
            base_accs = all_results.get((dn, baseline), [])
            if len(smm_accs) >= 5 and len(base_accs) >= 5:
                try:
                    stat, p_val = wilcoxon(smm_accs, base_accs, alternative='greater')
                    sig = "✅ SIGNIFICANT" if p_val < 0.05 else "❌ Not significant"
                    print(f"    SMM-MOGA vs {baseline}: p={p_val:.4f} {sig}")
                except Exception as e:
                    print(f"    SMM-MOGA vs {baseline}: Could not compute ({e})")

    # ══════════════════════════════════════════════════════════════
    # 7. RUNTIME TABLE
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  RUNTIME (seconds for 5 runs)")
    print("="*70)
    header2 = f"{'Method':<18}"
    for dn in dataset_names:
        header2 += f" | {dn:^22}"
    print(header2)
    print("-" * len(header2))
    for method in METHODS:
        row = f"{method:<18}"
        for dn in dataset_names:
            t = timing.get((dn, method), 0)
            row += f" | {t:>10.0f}s           "
        print(row)

    # ══════════════════════════════════════════════════════════════
    # 8. PLOTS
    # ══════════════════════════════════════════════════════════════
    print("\n📊 Generating plots...")

    # ── Plot 1: Grouped Bar Chart (main results) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dataset_names))
    width = 0.25
    colors = ['#B4B2A9', '#85B7EB', '#5DCAA5']

    for i, method in enumerate(METHODS):
        means = [np.mean(all_results.get((dn, method), [0])) for dn in dataset_names]
        stds  = [np.std(all_results.get((dn, method), [0]))  for dn in dataset_names]
        bars = ax.bar(x + i*width, means, width, yerr=stds, label=method,
                      color=colors[i], edgecolor='white', capsize=4, linewidth=0.5)
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Test Accuracy (mean ± std)', fontsize=12)
    ax.set_title('SMM-MOGA vs Baselines: 3 Datasets × 5 Runs', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(dataset_names)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("benchmark_comparison.png", dpi=300)
    print("  Saved: benchmark_comparison.png")

    # ── Plot 2: Box Plot per dataset ──
    fig, axes = plt.subplots(1, len(dataset_names), figsize=(5*len(dataset_names), 5))
    if len(dataset_names) == 1:
        axes = [axes]

    for idx, dn in enumerate(dataset_names):
        ax = axes[idx]
        data = [all_results.get((dn, m), []) for m in METHODS]
        bp = ax.boxplot(data, labels=METHODS, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(dn, fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.suptitle('Accuracy Distribution (5 Runs per Method)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("benchmark_boxplot.png", dpi=300)
    print("  Saved: benchmark_boxplot.png")

    # ── Save results to CSV ──
    rows = []
    for dn in dataset_names:
        for method in METHODS:
            accs = all_results.get((dn, method), [])
            rows.append({
                "Dataset": dn,
                "Method": method,
                "Mean_Accuracy": np.mean(accs) if accs else None,
                "Std_Accuracy": np.std(accs) if accs else None,
                "Run_1": accs[0] if len(accs) > 0 else None,
                "Run_2": accs[1] if len(accs) > 1 else None,
                "Run_3": accs[2] if len(accs) > 2 else None,
                "Run_4": accs[3] if len(accs) > 3 else None,
                "Run_5": accs[4] if len(accs) > 4 else None,
                "Total_Time_s": timing.get((dn, method), None),
            })
    df_results = pd.DataFrame(rows)
    df_results.to_csv("benchmark_results.csv", index=False)
    print("  Saved: benchmark_results.csv")

    # ══════════════════════════════════════════════════════════════
    # 9. FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  BENCHMARK COMPLETE!")
    print("="*70)
    print(f"  Total experiments: {len(all_results)} method-dataset combinations")
    print(f"  Total runs: {sum(len(v) for v in all_results.values())}")
    total_time = sum(timing.values())
    print(f"  Total time: {total_time/3600:.1f} hours ({total_time:.0f}s)")
    print("\n  Output files:")
    print("    benchmark_comparison.png  — Grouped bar chart")
    print("    benchmark_boxplot.png     — Box plots per dataset")
    print("    benchmark_results.csv     — Raw results for paper")
    print("="*70)
