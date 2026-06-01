"""
╔══════════════════════════════════════════════════════════════════════╗
║   SMM-MOGA — COMPLETE FINAL VERSION                                  ║
║   3 Datasets × 4 Methods × 5 Runs = 60 Experiments                  ║
║                                                                      ║
║   ALL FIXES + ALL PLOTS + KAGGLE KEEP-ALIVE + CHECKPOINTING         ║
║                                                                      ║
║   FIXES:                                                             ║
║    Train(60%) / Val(20%) / Test(20%) — test used ONCE             ║
║    True Pareto MOGA (NSGA-II dominance + crowding distance)       ║
║    Ablation Study (4 variants)                                     ║
║    optimizer_type as 6th gene → 1,728 search space                ║
║    F1-Macro as primary metric                                      ║
║    Zero data leakage preprocessing                                 ║
║                                                                      ║
║   PLOTS (7 total):                                                   ║
║    ablation_f1_comparison.png — Grouped bar chart                 ║
║    ablation_stepup.png        — Per-dataset ablation bars         ║
║    ablation_boxplot.png       — Box plots (test F1)               ║
║    pareto_front.png           — F1 vs Params Pareto front         ║
║    shap_importance.png        — Gene importance over gens         ║
║    convergence_curve.png      — Best F1 per generation            ║
║   📄 results_final.csv          — Raw results                       ║
║                                                                      ║
║   KAGGLE:                                                            ║
║    Keep-alive thread (prints every 4 min → prevents timeout)      ║
║    JSON checkpoint after every run (auto-resume on crash)         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import random
import time
import threading
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# KAGGLE KEEP-ALIVE — prints a dot every 4 min to prevent timeout
# ══════════════════════════════════════════════════════════════
_keep_alive_running = True

def _keep_alive_worker():
    counter = 0
    while _keep_alive_running:
        time.sleep(240)   # every 4 minutes
        counter += 1
        print(f"  ⏳ Still running... heartbeat #{counter} "
              f"({counter*4} min elapsed)", flush=True)

keep_alive_thread = threading.Thread(target=_keep_alive_worker, daemon=True)
keep_alive_thread.start()



# CHECKPOINT HELPERS
CHECKPOINT_FILE = "smm_moga_checkpoint.json"

def save_checkpoint(all_f1s, all_accs, timing, completed_runs):
    """Save progress after every run so we can resume on crash."""
    data = {
        "all_f1s":       {str(k): v for k, v in all_f1s.items()},
        "all_accs":      {str(k): v for k, v in all_accs.items()},
        "timing":        {str(k): v for k, v in timing.items()},
        "completed_runs": completed_runs,
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  💾 Checkpoint saved → {CHECKPOINT_FILE}", flush=True)

def load_checkpoint():
    """Load checkpoint if exists, return None otherwise."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None, None, None, []
    with open(CHECKPOINT_FILE) as f:
        data = json.load(f)
    all_f1s  = {eval(k): v for k, v in data["all_f1s"].items()}
    all_accs = {eval(k): v for k, v in data["all_accs"].items()}
    timing   = {eval(k): v for k, v in data["timing"].items()}
    completed = data.get("completed_runs", [])
    print(f"  🔄 Checkpoint loaded — {len(completed)} runs already done")
    return all_f1s, all_accs, timing, completed


# CONFIGURATION
GENE_NAMES = [
    "neurons_layer1", "neurons_layer2", "dropout_rate",
    "learning_rate",  "batch_size",     "optimizer_type",
]
GENE_CHOICES = [
    [16, 32, 64, 128],
    [16, 32, 64, 128],
    [0.0, 0.1, 0.2, 0.3],
    [0.001, 0.005, 0.01],
    [16, 32, 64],
    ["adam", "sgd", "rmsprop"],
]
OPTIMIZER_ENCODING = {"adam": 0, "sgd": 1, "rmsprop": 2}

POP_SIZE         = 20
GENERATIONS      = 15
MUTATION_RATE    = 0.4
EPOCHS_GA        = 30
META_MIN_SAMPLES = 20
NUM_RUNS         = 5
METHODS = ["Random Search", "Standard GA", "GA+Surrogate", "SMM-MOGA"]

# Global storage for extra plot data (collected from SMM-MOGA runs)
pareto_data     = {}   # {dataset_name: [(f1, params), ...]}
shap_hist_data  = {}   # {dataset_name: [importance_array per gen, ...]}
convergence_data = {}  # {dataset_name: [best_f1 per gen, ...]}


# 1. DATA LOADING

def load_brfss(path="BRFSS1998.csv"):
    print(f"  Loading BRFSS from {path}...")
    df = pd.read_csv(path)
    df["risk_score"] = (
        (df["bphigh"] == 1).astype(int) + (df["smoke100"] == 1).astype(int) +
        (df["exerany"] == 2).astype(int) + (df["hlthplan"] == 2).astype(int)
    )
    df["label"] = df["risk_score"].apply(
        lambda s: "Unhealthy" if s >= 3 else ("At_Risk" if s == 2 else "Healthy"))
    df.drop(columns=["risk_score"], inplace=True)

    feature_names = ["age", "_bmi", "checkup", "sex", "drinkany", "educa",
                     "income2", "physhlth", "menthlth", "genhlth",
                     "diabetes", "bloodcho", "seatbelt", "flushot"]
    df[feature_names] = df[feature_names].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=feature_names, inplace=True)

    X  = df[feature_names].values.astype(float)
    le = LabelEncoder()
    y  = le.fit_transform(df["label"])
    print(f"    Shape: {X.shape} | Classes: {list(le.classes_)}")
    print(f"    Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, len(le.classes_), feature_names, "BRFSS 1998"


def load_diabetes(path="diabetes.csv"):
    print(f"  Loading Diabetes from {path}...")
    df = pd.read_csv(path)
    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[col] = df[col].replace(0, np.nan)
    X = df[feature_names].values.astype(float)
    y = df["Outcome"].values
    print(f"    Shape: {X.shape} | Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, 2, feature_names, "Pima Diabetes"


def load_heart():
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "heart-disease/processed.cleveland.data")
    cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal','target']
    print("  Loading Heart Disease from UCI...")
    df = pd.read_csv(url, names=cols, na_values='?')
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    feature_names = cols[:-1]
    df[feature_names] = df[feature_names].apply(pd.to_numeric, errors='coerce')
    X = df[feature_names].values.astype(float)
    y = df['target'].values
    print(f"    Shape: {X.shape} | (0={sum(y==0)}, 1={sum(y==1)})")
    return X, y, 2, feature_names, "UCI Heart Disease"


# 2. PREPROCESSING — all fit on train only, zero leakage

def iqr_cap_fit(X):
    caps, X_out = [], X.copy()
    for col in range(X.shape[1]):
        Q1, Q3 = np.percentile(X[:, col], [25, 75])
        lo, hi = Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1)
        caps.append((lo, hi))
        X_out[:, col] = np.clip(X[:, col], lo, hi)
    return X_out, caps

def iqr_cap_transform(X, caps):
    X_out = X.copy()
    for col, (lo, hi) in enumerate(caps):
        X_out[:, col] = np.clip(X[:, col], lo, hi)
    return X_out

def apply_smote(X, y, seed):
    _, counts = np.unique(y, return_counts=True)
    k = min(5, counts.min() - 1)
    if k < 1: return X, y
    return SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)

def preprocess_pipeline(X_tr, X_va, X_te, y_tr, dataset_name, seed):
    if np.isnan(X_tr).any() or np.isnan(X_va).any() or np.isnan(X_te).any():
        imp  = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_va = imp.transform(X_va)
        X_te = imp.transform(X_te)
    if "BRFSS" in dataset_name:
        X_tr, caps = iqr_cap_fit(X_tr)
        X_va = iqr_cap_transform(X_va, caps)
        X_te = iqr_cap_transform(X_te, caps)
    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_va = sc.transform(X_va)
    X_te = sc.transform(X_te)
    X_tr, y_tr = apply_smote(X_tr, y_tr, seed)
    return X_tr, X_va, X_te, y_tr


# 3. CORE HELPERS

def chromosome_to_vector(chrom):
    return [float(OPTIMIZER_ENCODING[g]) if isinstance(g, str)
            else float(g) for g in chrom]

def random_chromosome():
    return [random.choice(c) for c in GENE_CHOICES]

def build_mlp(chrom, n_features, n_classes):
    n1, n2, drop, lr, _, opt_name = chrom
    opt_map = {
        "adam":    tf.keras.optimizers.Adam(float(lr)),
        "sgd":     tf.keras.optimizers.SGD(float(lr), momentum=0.9),
        "rmsprop": tf.keras.optimizers.RMSprop(float(lr)),
    }
    m = Sequential([
        Dense(int(n1), activation="relu", input_shape=(n_features,)),
        Dropout(float(drop)),
        Dense(int(n2), activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer=opt_map[opt_name],
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m

def evaluate_on_val(chrom, X_tr, y_tr_cat, X_va, y_va, n_classes):
    """Train on X_tr, measure F1 on X_va only. Test never touched."""
    _, _, _, _, batch_size, _ = chrom
    model  = build_mlp(chrom, X_tr.shape[1], n_classes)
    params = model.count_params()
    model.fit(X_tr, y_tr_cat, epochs=EPOCHS_GA, batch_size=int(batch_size),
              verbose=0, validation_split=0.1,
              callbacks=[EarlyStopping(monitor='val_loss', patience=3,
                                       restore_best_weights=True)])
    preds  = np.argmax(model.predict(X_va, verbose=0), axis=1)
    f1_val = f1_score(y_va, preds, average='macro', zero_division=0)
    tf.keras.backend.clear_session()
    return f1_val, params

def evaluate_on_test(chrom, X_tr, y_tr_cat, X_va, y_va_cat, X_te, y_te, n_classes):
    """Retrain on train+val, evaluate ONCE on test. ✅"""
    _, _, _, _, batch_size, _ = chrom
    X_full = np.vstack([X_tr, X_va])
    y_full = np.vstack([y_tr_cat, y_va_cat])
    model  = build_mlp(chrom, X_full.shape[1], n_classes)
    model.fit(X_full, y_full, epochs=EPOCHS_GA, batch_size=int(batch_size),
              verbose=0,
              callbacks=[EarlyStopping(monitor='loss', patience=3,
                                       restore_best_weights=True)])
    preds = np.argmax(model.predict(X_te, verbose=0), axis=1)
    f1    = f1_score(y_te, preds, average='macro', zero_division=0)
    acc   = accuracy_score(y_te, preds)
    tf.keras.backend.clear_session()
    return f1, acc, model.count_params()


# 4. PARETO HELPERS (NSGA-II style)

def dominates(a, b):
    """a dominates b: better or equal in all, strictly better in one."""
    return (a[0] >= b[0] and a[1] <= b[1]) and \
           (a[0] > b[0]  or  a[1] < b[1])

def non_dominated_sort(objs):
    n  = len(objs)
    dc = [0] * n
    ds = [[] for _ in range(n)]
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if dominates(objs[i], objs[j]):  ds[i].append(j)
            elif dominates(objs[j], objs[i]): dc[i] += 1
        if dc[i] == 0: fronts[0].append(i)
    cur = 0
    while fronts[cur]:
        nxt = []
        for i in fronts[cur]:
            for j in ds[i]:
                dc[j] -= 1
                if dc[j] == 0: nxt.append(j)
        cur += 1; fronts.append(nxt)
    return [f for f in fronts if f]

def crowding_distance(front_idx, objs):
    n = len(front_idx)
    if n <= 2: return [float('inf')] * n
    dist = [0.0] * n
    for obj in range(2):
        vals = sorted([(objs[front_idx[i]][obj], i) for i in range(n)])
        dist[vals[0][1]] = dist[vals[-1][1]] = float('inf')
        rng = vals[-1][0] - vals[0][0]
        if rng == 0: continue
        for k in range(1, n-1):
            dist[vals[k][1]] += (vals[k+1][0] - vals[k-1][0]) / rng
    return dist

def pareto_select(scored_pop, n_select):
    objs   = [(s[1], -s[2]) for s in scored_pop]
    fronts = non_dominated_sort(objs)
    sel    = []
    for front in fronts:
        if len(sel) + len(front) <= n_select:
            sel.extend(front)
        else:
            cd  = crowding_distance(front, objs)
            top = sorted(zip(cd, front), reverse=True)
            sel.extend(idx for _, idx in top[:n_select - len(sel)])
            break
    return [scored_pop[i][0] for i in sel]


# 5. SURROGATE + SHAP

def build_surrogate(history):
    X_m = np.array([h[0] for h in history])
    y_m = np.array([h[1] for h in history])
    rf  = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_m, y_m)
    return rf

def compute_shap_importance(surrogate, history):
    X_m = np.array([h[0] for h in history])
    sv  = shap.TreeExplainer(surrogate).shap_values(X_m)
    return np.mean(np.abs(sv), axis=0)

def shap_guided_mutation(chrom, importance):
    child    = chrom.copy()
    inv      = 1.0 / (importance + 1e-8)
    probs    = inv / inv.sum()
    gene_idx = np.random.choice(len(child), p=probs)
    child[gene_idx] = random.choice(GENE_CHOICES[gene_idx])
    return child

def random_mutation(chrom):
    child    = chrom.copy()
    gene_idx = random.randint(0, len(chrom) - 1)
    child[gene_idx] = random.choice(GENE_CHOICES[gene_idx])
    return child

def uniform_crossover(p1, p2):
    return [random.choice([p1[i], p2[i]]) for i in range(len(p1))]


# 6. FOUR SEARCH METHODS

def run_random_search(X_tr, y_tr_cat, X_va, y_va, n_classes, seed, dataset_name=None):
    np.random.seed(seed); random.seed(seed); tf.random.set_seed(seed)
    best_chrom, best_f1 = None, -1
    for _ in range(POP_SIZE * GENERATIONS):
        chrom     = random_chromosome()
        f1_val, _ = evaluate_on_val(chrom, X_tr, y_tr_cat, X_va, y_va, n_classes)
        if f1_val > best_f1:
            best_f1, best_chrom = f1_val, chrom
    return best_chrom


def run_standard_ga(X_tr, y_tr_cat, X_va, y_va, n_classes, seed, dataset_name=None):
    np.random.seed(seed); random.seed(seed); tf.random.set_seed(seed)
    population = [random_chromosome() for _ in range(POP_SIZE)]
    best_chrom, best_f1 = None, -1
    for gen in range(GENERATIONS):
        scored = []
        for chrom in population:
            f1_val, params = evaluate_on_val(chrom, X_tr, y_tr_cat, X_va, y_va, n_classes)
            scored.append((chrom, f1_val, params))
            if f1_val > best_f1: best_f1, best_chrom = f1_val, chrom
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [c for c, _, _ in scored[:POP_SIZE // 2]]
        new_pop  = selected.copy()
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.sample(selected, 2)
            child  = uniform_crossover(p1, p2)
            if random.random() < MUTATION_RATE: child = random_mutation(child)
            new_pop.append(child)
        population = new_pop
    return best_chrom


def run_ga_surrogate(X_tr, y_tr_cat, X_va, y_va, n_classes, seed, dataset_name=None):
    """GA + surrogate, but RANDOM mutation — isolates SHAP contribution."""
    np.random.seed(seed); random.seed(seed); tf.random.set_seed(seed)
    population = [random_chromosome() for _ in range(POP_SIZE)]
    history    = []
    best_chrom, best_f1 = None, -1
    for gen in range(GENERATIONS):
        scored = []
        for chrom in population:
            f1_val, params = evaluate_on_val(chrom, X_tr, y_tr_cat, X_va, y_va, n_classes)
            scored.append((chrom, f1_val, params))
            history.append((chromosome_to_vector(chrom), f1_val))
            if f1_val > best_f1: best_f1, best_chrom = f1_val, chrom
        if len(history) >= META_MIN_SAMPLES:
            build_surrogate(history)   # built but mutation stays random
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [c for c, _, _ in scored[:POP_SIZE // 2]]
        new_pop  = selected.copy()
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.sample(selected, 2)
            child  = uniform_crossover(p1, p2)
            if random.random() < MUTATION_RATE: child = random_mutation(child)
            new_pop.append(child)
        population = new_pop
    return best_chrom


def run_smm_moga(X_tr, y_tr_cat, X_va, y_va, n_classes, seed, dataset_name=None):
    """
    Full SMM-MOGA:
     Pareto selection (true MOGA)
     surrogate model
     SHAP-guided mutation
    Collects data for Pareto / SHAP / Convergence plots.
    """
    np.random.seed(seed); random.seed(seed); tf.random.set_seed(seed)
    population  = [random_chromosome() for _ in range(POP_SIZE)]
    history     = []
    surrogate   = None
    importance  = np.ones(len(GENE_NAMES)) / len(GENE_NAMES)
    best_chrom, best_f1 = None, -1

    # ── Data collectors for plots ──────────────────────────────────────
    all_solutions   = []   # (f1_val, params) for every evaluated architecture
    shap_per_gen    = []   # importance vector per generation
    best_f1_per_gen = []   # convergence curve

    for gen in range(GENERATIONS):
        scored = []
        for chrom in population:
            f1_val, params = evaluate_on_val(
                chrom, X_tr, y_tr_cat, X_va, y_va, n_classes)
            scored.append((chrom, f1_val, params))
            history.append((chromosome_to_vector(chrom), f1_val))
            all_solutions.append((f1_val, params))
            if f1_val > best_f1: best_f1, best_chrom = f1_val, chrom

        best_f1_per_gen.append(best_f1)

        if len(history) >= META_MIN_SAMPLES:
            surrogate  = build_surrogate(history)
            importance = compute_shap_importance(surrogate, history)

        shap_per_gen.append(importance.copy())

        # Pareto selection ✅
        selected = pareto_select(scored, POP_SIZE // 2)
        new_pop  = selected.copy()
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.sample(selected, 2)
            child  = uniform_crossover(p1, p2)
            if random.random() < MUTATION_RATE:
                child = (shap_guided_mutation(child, importance)
                         if surrogate is not None else random_mutation(child))
            new_pop.append(child)
        population = new_pop

    # ── Store plot data (accumulate across runs) ───────────────────────
    if dataset_name:
        if dataset_name not in pareto_data:
            pareto_data[dataset_name]      = []
            shap_hist_data[dataset_name]   = []
            convergence_data[dataset_name] = []
        pareto_data[dataset_name].extend(all_solutions)
        shap_hist_data[dataset_name].append(np.array(shap_per_gen))
        convergence_data[dataset_name].append(best_f1_per_gen)

    return best_chrom


METHOD_FUNCS = {
    "Random Search": run_random_search,
    "Standard GA":   run_standard_ga,
    "GA+Surrogate":  run_ga_surrogate,
    "SMM-MOGA":      run_smm_moga,
}


# 7. PLOTTING FUNCTIONS

def plot_pareto_front(dataset_names):
    """
    Pareto Front: F1-Macro vs Parameter Count.
    All evaluated solutions shown as grey dots.
    Pareto-optimal solutions highlighted in red.
    Best solution (highest F1 on Pareto front) shown as gold star.
    """
    n_ds  = len(dataset_names)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5))
    if n_ds == 1: axes = [axes]

    for idx, dn in enumerate(dataset_names):
        ax       = axes[idx]
        sols     = pareto_data.get(dn, [])
        if not sols:
            ax.set_title(f'{dn}\n(no data)'); continue

        f1s    = np.array([s[0] for s in sols])
        params = np.array([s[1] for s in sols])

        # Identify Pareto front
        objs       = list(zip(f1s, -params))
        fronts     = non_dominated_sort(objs)
        pareto_idx = set(fronts[0]) if fronts else set()

        p_f1s  = f1s[list(pareto_idx)]
        p_par  = params[list(pareto_idx)]
        others = [i for i in range(len(sols)) if i not in pareto_idx]

        # Best on Pareto front = highest F1
        best_i = list(pareto_idx)[np.argmax(p_f1s)]

        ax.scatter(params[others], f1s[others],
                   alpha=0.25, color='#AAAAAA', s=15, label='All solutions')
        ax.scatter(p_par, p_f1s,
                   color='#E74C3C', s=50, zorder=3, label='Pareto front')
        ax.scatter(params[best_i], f1s[best_i],
                   color='gold', s=200, marker='*', zorder=4,
                   edgecolors='black', linewidth=0.8,
                   label=f'Best (F1={f1s[best_i]:.3f}, P={params[best_i]})')

        # Draw Pareto frontier line
        sorted_p = sorted(zip(p_par, p_f1s))
        ax.step([x for x, _ in sorted_p], [y for _, y in sorted_p],
                where='post', color='#E74C3C', alpha=0.5, linewidth=1.5,
                linestyle='--')

        ax.set_xlabel('Number of Parameters (Model Complexity)', fontsize=11)
        ax.set_ylabel('F1-Macro (Validation)', fontsize=11)
        ax.set_title(f'Pareto Front — {dn}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle('SMM-MOGA: Accuracy–Compactness Trade-off (Pareto Front)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("pareto_front.png", dpi=300)
    plt.close()
    print("  Saved: pareto_front.png")


def plot_shap_importance(dataset_names):
    """
    SHAP Gene Importance over Generations.
    Each line = one gene. Shows how importance evolves as GA converges.
    """
    n_ds  = len(dataset_names)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5))
    if n_ds == 1: axes = [axes]

    gene_colors = ['#E74C3C', '#3498DB', '#2ECC71',
                   '#F39C12', '#9B59B6', '#1ABC9C']

    for idx, dn in enumerate(dataset_names):
        ax   = axes[idx]
        data = shap_hist_data.get(dn, [])
        if not data:
            ax.set_title(f'{dn}\n(no data)'); continue

        # Average importance across all runs
        avg_importance = np.mean(np.array(data), axis=0)  # shape: (gens, n_genes)
        gens = np.arange(1, avg_importance.shape[0] + 1)

        for g_idx, (gene_name, color) in enumerate(zip(GENE_NAMES, gene_colors)):
            ax.plot(gens, avg_importance[:, g_idx],
                    marker='o', markersize=4, linewidth=2,
                    color=color, label=gene_name)

        # Mark when surrogate activates
        ax.axvline(x=META_MIN_SAMPLES / POP_SIZE + 1, color='gray',
                   linestyle='--', alpha=0.7, label='Surrogate activates')

        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('Mean |SHAP Value|', fontsize=11)
        ax.set_title(f'Gene Importance — {dn}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=0.3)

    plt.suptitle('SHAP Gene Importance Dynamics (SMM-MOGA Meta-Model)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_importance.png", dpi=300)
    plt.close()
    print("  Saved: shap_importance.png")


def plot_convergence(dataset_names, all_f1s_rs, all_f1s_ga):
    """
    Convergence Curve: Best F1 per generation (SMM-MOGA vs baselines).
    Shows SMM-MOGA converges faster / to higher value.
    """
    n_ds  = len(dataset_names)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5))
    if n_ds == 1: axes = [axes]

    for idx, dn in enumerate(dataset_names):
        ax   = axes[idx]
        data = convergence_data.get(dn, [])
        if not data:
            ax.set_title(f'{dn}\n(no data)'); continue

        # SMM-MOGA convergence (mean ± std across runs)
        conv_arr = np.array(data)   # shape: (n_runs, n_gens)
        mean_c   = conv_arr.mean(axis=0)
        std_c    = conv_arr.std(axis=0)
        gens     = np.arange(1, len(mean_c) + 1)

        ax.plot(gens, mean_c, color='#5DCAA5', linewidth=2.5,
                marker='o', markersize=4, label='SMM-MOGA')
        ax.fill_between(gens, mean_c - std_c, mean_c + std_c,
                        alpha=0.2, color='#5DCAA5')

        # Add final performance lines for other methods
        for method, color, style in [
            ("Random Search", '#B4B2A9', '--'),
            ("Standard GA",   '#85B7EB', '-.'),
            ("GA+Surrogate",  '#F5A623', ':'),
        ]:
            vals = all_f1s_rs.get((dn, method), [])
            if vals:
                ax.axhline(y=np.mean(vals), color=color,
                           linestyle=style, linewidth=1.5,
                           label=f'{method} (final avg)')

        ax.axvline(x=META_MIN_SAMPLES / POP_SIZE + 1, color='gray',
                   linestyle='--', alpha=0.6, label='Surrogate activates')

        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('Best F1-Macro (Validation)', fontsize=11)
        ax.set_title(f'Convergence — {dn}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle('SMM-MOGA Convergence vs Baselines',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("convergence_curve.png", dpi=300)
    plt.close()
    print("  Saved: convergence_curve.png")


# ══════════════════════════════════════════════════════════════
# 8. MAIN BENCHMARK LOOP
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 70)
    print("  SMM-MOGA — COMPLETE FINAL VERSION")
    print("  Split: 60% Train | 20% Val | 20% Test (used once)")
    print(f"  {len(METHODS)} Methods × 3 Datasets × {NUM_RUNS} Runs = "
          f"{len(METHODS)*3*NUM_RUNS} Experiments")
    print("=" * 70)

    # ── Load checkpoint if exists ──────────────────────────────────────
    all_f1s, all_accs, timing, completed_runs = load_checkpoint()
    if all_f1s is None:
        all_f1s, all_accs, timing, completed_runs = {}, {}, {}, []

    # ── Load datasets ──────────────────────────────────────────────────
    print("\n📦 Loading datasets...")
    datasets = []
    try:    datasets.append(load_brfss("BRFSS1998.csv"))
    except FileNotFoundError: print("  ⚠️  BRFSS1998.csv not found")
    try:    datasets.append(load_diabetes("diabetes.csv"))
    except FileNotFoundError: print("  ⚠️  diabetes.csv not found")
    try:    datasets.append(load_heart())
    except Exception as e: print(f"  ⚠️  Heart load failed: {e}")

    if not datasets: print("❌ No datasets found!"); exit(1)

    # ── Main loop ──────────────────────────────────────────────────────
    for X, y, n_classes, feature_names, dataset_name in datasets:
        print(f"\n{'═'*70}\n  DATASET: {dataset_name}\n{'═'*70}")

        for method_name in METHODS:
            run_key = f"{dataset_name}_{method_name}"

            # Skip if already done (checkpoint resume)
            completed_count = completed_runs.count(run_key)
            if completed_count >= NUM_RUNS:
                print(f"  ⏭  {method_name} on {dataset_name} — already done, skipping")
                continue

            print(f"\n  ── Method: {method_name} ──")

            # Initialize lists (might have partial results from checkpoint)
            if (dataset_name, method_name) not in all_f1s:
                all_f1s[(dataset_name, method_name)]  = []
                all_accs[(dataset_name, method_name)] = []
                timing[(dataset_name, method_name)]   = 0.0

            t0 = time.time()

            for run in range(completed_count, NUM_RUNS):
                seed = 42 + run * 10

                # 1. Split
                X_dev, X_te, y_dev, y_te = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=seed)
                X_tr, X_va, y_tr, y_va = train_test_split(
                    X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=seed)

                # 2. Preprocess (no leakage)
                X_tr, X_va, X_te_p, y_tr = preprocess_pipeline(
                    X_tr, X_va, X_te.copy(), y_tr, dataset_name, seed)

                # 3. Encode
                y_tr_cat = to_categorical(y_tr, num_classes=n_classes)
                y_va_cat = to_categorical(y_va, num_classes=n_classes)

                print(f"    Run {run+1}/{NUM_RUNS} (seed={seed}) "
                      f"[tr={len(y_tr)} va={len(y_va)} te={len(y_te)}]...",
                      end=" ", flush=True)

                # 4. Search on validation
                best_chrom = METHOD_FUNCS[method_name](
                    X_tr, y_tr_cat, X_va, y_va, n_classes, seed,
                    dataset_name=(dataset_name if method_name == "SMM-MOGA" else None))

                # 5. Final test evaluation (once!)
                f1_te, acc_te, params = evaluate_on_test(
                    best_chrom, X_tr, y_tr_cat, X_va, y_va_cat,
                    X_te_p, y_te, n_classes)

                all_f1s[(dataset_name, method_name)].append(f1_te)
                all_accs[(dataset_name, method_name)].append(acc_te)
                completed_runs.append(run_key)
                print(f"F1(test)={f1_te:.4f} | Acc={acc_te:.4f} | Params={params}")

                # Save checkpoint after every single run
                save_checkpoint(all_f1s, all_accs, timing, completed_runs)

            elapsed = time.time() - t0
            timing[(dataset_name, method_name)] += elapsed

            f1s  = all_f1s[(dataset_name, method_name)]
            accs = all_accs[(dataset_name, method_name)]
            print(f"    ✅ {method_name}: "
                  f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
                  f"Acc={np.mean(accs):.4f}±{np.std(accs):.4f}")

    dataset_names = [d[4] for d in datasets]

    # 9. RESULTS TABLES
    # ══════════════════════════════════════════════════════════════
    def print_table(title, data_dict):
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
        header = f"{'Method':<18}"
        for dn in dataset_names: header += f" | {dn:^24}"
        print(header); print("-"*len(header))
        for method in METHODS:
            row = f"{method:<18}"
            for dn in dataset_names:
                v = data_dict.get((dn, method), [])
                row += (f" | {np.mean(v):.4f} ± {np.std(v):.4f}    "
                        if v else f" | {'N/A':^24}")
            print(row)

    print_table("F1-MACRO — TEST SET (Primary Metric)", all_f1s)
    print_table("ACCURACY — TEST SET (Reference)",      all_accs)

    # ── Ablation table ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  ABLATION STUDY — Component Contribution")
    print("="*70)
    for dn in dataset_names:
        print(f"\n  {dn}:")
        prev = None
        for method in METHODS:
            v    = all_f1s.get((dn, method), [])
            mean = np.mean(v) if v else 0
            gain = f"  Δ=+{mean-prev:.4f}" if prev is not None else ""
            print(f"    {method:<20} F1={mean:.4f} ± {np.std(v) if v else 0:.4f}{gain}")
            prev = mean

    # ── Wilcoxon ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STATISTICAL SIGNIFICANCE — Wilcoxon (F1, test set)")
    print("="*70)
    for dn in dataset_names:
        print(f"\n  {dn}:")
        smm = all_f1s.get((dn, "SMM-MOGA"), [])
        for bl in ["Random Search", "Standard GA", "GA+Surrogate"]:
            base = all_f1s.get((dn, bl), [])
            if len(smm) >= 5 and len(base) >= 5:
                try:
                    _, p = wilcoxon(smm, base, alternative='greater')
                    print(f"    vs {bl:<15}: p={p:.4f} "
                          f"{'✅ SIG' if p < 0.05 else '❌ Not sig'}")
                except Exception as e:
                    print(f"    vs {bl}: ({e})")

    # 10. ALL PLOTS
    # ══════════════════════════════════════════════════════════════
    print("\n📊 Generating all plots...")
    colors = ['#B4B2A9', '#85B7EB', '#F5A623', '#5DCAA5']

    # ── Plot 1: Ablation bar chart ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    x, w = np.arange(len(dataset_names)), 0.2
    for i, method in enumerate(METHODS):
        means = [np.mean(all_f1s.get((dn, method), [0])) for dn in dataset_names]
        stds  = [np.std(all_f1s.get((dn, method), [0]))  for dn in dataset_names]
        bars  = ax.bar(x + i*w, means, w, yerr=stds, label=method,
                       color=colors[i], edgecolor='white', capsize=4)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('F1-Macro on Test Set (mean ± std)', fontsize=12)
    ax.set_title('Ablation Study — SMM-MOGA vs Baselines', fontsize=13, fontweight='bold')
    ax.set_xticks(x + w*1.5); ax.set_xticklabels(dataset_names)
    ax.legend(fontsize=9); ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig("ablation_f1_comparison.png", dpi=300); plt.close()
    print("  Saved: ablation_f1_comparison.png")

    # ── Plot 2: Step-up per dataset ────────────────────────────────────
    fig, axes = plt.subplots(1, len(dataset_names), figsize=(5*len(dataset_names), 5))
    if len(dataset_names) == 1: axes = [axes]
    for idx, dn in enumerate(dataset_names):
        ax    = axes[idx]
        means = [np.mean(all_f1s.get((dn, m), [0])) for m in METHODS]
        stds  = [np.std(all_f1s.get((dn, m), [0]))  for m in METHODS]
        bars  = ax.bar(METHODS, means, yerr=stds, color=colors, edgecolor='white', capsize=4)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_title(dn, fontsize=11, fontweight='bold')
        ax.set_ylabel('F1-Macro (Test)')
        ax.set_xticklabels(METHODS, rotation=15, ha='right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.suptitle("Ablation: Each Component's Contribution", fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig("ablation_stepup.png", dpi=300); plt.close()
    print("  Saved: ablation_stepup.png")

    # ── Plot 3: Box plots ──────────────────────────────────────────────
    fig, axes2 = plt.subplots(1, len(dataset_names), figsize=(5*len(dataset_names), 5))
    if len(dataset_names) == 1: axes2 = [axes2]
    for idx, dn in enumerate(dataset_names):
        ax   = axes2[idx]
        data = [all_f1s.get((dn, m), []) for m in METHODS]
        bp   = ax.boxplot(data, labels=METHODS, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_title(dn, fontsize=11, fontweight='bold')
        ax.set_ylabel('F1-Macro (Test)')
        ax.set_xticklabels(METHODS, rotation=15, ha='right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.suptitle('F1-Macro Distribution — Test Set (5 Runs)', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig("ablation_boxplot.png", dpi=300); plt.close()
    print("  Saved: ablation_boxplot.png")

    # ── Plot 4: Pareto Front  ───────────────────────────────────
    plot_pareto_front(dataset_names)

    # ── Plot 5: SHAP Gene Importance  ──────────────────────────
    plot_shap_importance(dataset_names)

    # ── Plot 6: Convergence Curve  ──────────────────────────────
    plot_convergence(dataset_names, all_f1s, all_f1s)

    # ── CSV export ─────────────────────────────────────────────────────
    rows = []
    for dn in dataset_names:
        for method in METHODS:
            f1s  = all_f1s.get((dn, method), [])
            accs = all_accs.get((dn, method), [])
            rows.append({
                "Dataset": dn, "Method": method,
                "Mean_F1": np.mean(f1s) if f1s else None,
                "Std_F1":  np.std(f1s)  if f1s else None,
                "Mean_Acc": np.mean(accs) if accs else None,
                "Std_Acc":  np.std(accs)  if accs else None,
                **{f"F1_Run_{i+1}": (f1s[i] if i < len(f1s) else None) for i in range(5)},
                "Time_s": timing.get((dn, method), None),
            })
    pd.DataFrame(rows).to_csv("results_final.csv", index=False)
    print("  Saved: results_final.csv")

    # ── Stop keep-alive ────────────────────────────────────────────────
    _keep_alive_running = False

    print("\n" + "="*70)
    print("   BENCHMARK COMPLETE!")
    print("="*70)
    total = sum(timing.values())
    print(f"  Total time: {total/3600:.1f}h | Search space: 1,728 configs")
    print("\n  Output files (7 total):")
    print("    ablation_f1_comparison.png  — Grouped bar chart")
    print("    ablation_stepup.png         — Per-dataset ablation")
    print("    ablation_boxplot.png        — Box plots")
    print("    pareto_front.png            — F1 vs Params Pareto ")
    print("    shap_importance.png         — Gene importance over gens ")
    print("    convergence_curve.png       — Best F1 per generation ")
    print("    results_final.csv           — Raw data")
    print("="*70)
