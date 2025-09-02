import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

from scipy.stats import loguniform, randint
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# 1) Load Data
# ---------------------------
_t1 = time.perf_counter()

df = pd.read_csv("clean_train.csv")

# Fields that are unsafe (IDs/leakage/fairness)
df = df.drop(columns=[
    "ID",
    "Customer_ID",
    "Month",
    "Occupation",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Interest_Rate",
    "Changed_Credit_Limit",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Age"
]
)

# Split features/target
X = df.drop(columns=['Credit_Score'])
y = df['Credit_Score'].astype("category")

# The target needs to be a single column with the same number of observations as the feature data
print(X.shape)
print(y.shape)

# Backend check
dt = time.perf_counter() - _t1
print(f"✔ 1) Load Data - finished in {dt:.2f} s", flush=True)

# 2) Train / Test split 
# ---------------------------
_t2 = time.perf_counter()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Backend check
dt = time.perf_counter() - _t2
print(f"✔ 2) Train / Test split - finished in {dt:.2f} s", flush=True)

# 3) Feature Engineering (impute + encode + scale)
# ---------------------------
_t3 = time.perf_counter()

numeric_selector = selector(dtype_include=np.number)
categorical_selector = selector(dtype_exclude=np.number)

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_selector),
        ("cat", categorical_pipe, categorical_selector),
    ],
    remainder="drop"
)

# Backend check
dt = time.perf_counter() - _t3
print(f"✔ 3) Feature Engineering - finished in {dt:.2f} s", flush=True)

# 4) Model Pipelines
# ---------------------------
_t4 = time.perf_counter()

pipe_lr = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=800, random_state=42)) 
])

pipe_rf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

# Backend check
dt = time.perf_counter() - _t4
print(f"✔ 4) Model Pipelines - finished in {dt:.2f} s", flush=True)

# 5) Hyperparameter Grids
# ---------------------------
_t5 = time.perf_counter()

param_grid_lr = { 
    "model__C": loguniform(1e-3, 1e2), 
    "model__class_weight": [None, "balanced"] 
}

param_grid_rf = {    
    "model__n_estimators": randint(200, 401),
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": randint(2, 6),
    "model__min_samples_leaf": randint(1, 3),
    "model__max_features": ["sqrt"],
    "model__class_weight": [None, "balanced"],
}

# Backend check
dt = time.perf_counter() - _t5
print(f"✔ 5) Hyperparameter Grids - finished in {dt:.2f} s", flush=True)

# 6) Cross-Validation + Search
# ---------------------------
_t6 = time.perf_counter()

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scoring = "f1_macro"

def tune_and_eval(pipe, param_dist, name, n_iter):
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=1,           
        pre_dispatch=1,
        random_state=42,
        refit=True,
        return_train_score=False,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_est = search.best_estimator_
    best_cv = search.best_score_

    # Test-set predictions
    y_pred = best_est.predict(X_test)

    # Metrics (macro for multi-class fairness; add weighted if you like)
    metrics = {
        "model": name,
        "best_cv_f1_macro": best_cv,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "test_recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
        "best_params": search.best_params_
    }

    # Print detailed report
    print(f"\n=== {name} — Best Params ===\n")
    print(search.best_params_)
    print(f"CV mean F1 (macro): {best_cv:.4f}")

    print(f"\n=== {name} — Test Classification Report ===\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix (normalized for readability)
    blue_matrix = LinearSegmentedColormap.from_list(
    "capco_blue",
    ["#D9EFFF", "#005D9F"],  # light → dark
    N=256
    )

    cm = confusion_matrix(y_test, y_pred, labels=best_est.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_est.classes_)
    disp.plot(xticks_rotation=45, values_format="d", cmap=blue_matrix)
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return metrics

results = []
results.append(tune_and_eval(pipe_lr, param_grid_lr, "LogisticRegression", n_iter=10))
results.append(tune_and_eval(pipe_rf, param_grid_rf, "RandomForest",       n_iter=10))

# Backend check
dt = time.perf_counter() - _t6
print(f"✔ 6) Cross-Validation + Search - finished in {dt:.2f} s", flush=True)

# 7) Results Table
# ---------------------------
_t7 = time.perf_counter()

results_df = pd.DataFrame(results)[[
    "model", "best_cv_f1_macro",
    "test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro",
    "best_params"
]].sort_values(by="test_f1_macro", ascending=False)

print(f"\n=== Model Comparison Table ===\n")
print(results_df.to_string(index=False))

print(f"\n=== Model Comparison Table ===\n")
print(results_df)

# Backend check
dt = time.perf_counter() - _t7
print(f"✔ 7) Results Table - finished in {dt:.2f} s", flush=True)
