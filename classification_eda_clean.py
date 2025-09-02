from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import seaborn as sns
import re

# 1) Load Data
# ---------------------------

DATA_DIR = Path(r"..") / "ADS" / "Datasets"
df_train = pd.read_csv(DATA_DIR / "Classification_train.csv")
df_test  = pd.read_csv(DATA_DIR / "Classification_test.csv")

# 2) Quick EDA
# ---------------------------

print("\n=== Shapes ===\n")
print("train:", df_train.shape, "\ntest: ", df_test.shape)

print("\n=== Missing Column(s) ===\n")
diff_cols = [c for c in df_train.columns if c not in df_test.columns]
if len(diff_cols) > 0:
    target = diff_cols
print(f"{target}")

print("\n=== Sample rows (Classification_train) ===\n")
print(df_train.head(3))

print("\n=== Info ===\n")
print(df_train.info())

print("\n=== Describe (numeric) ===\n")
print(df_train.describe().T)

print("\n=== Describe (categorical) ===\n")
print(df_train.describe(include="object").T)

print("\n=== Duplicates ===\n")
print(df_train.duplicated().sum())

print("\n=== Missing Data ===\n")
mis = df_train.isna().sum()
mis = mis[mis > 0].sort_values(ascending=False)
missing_tbl = (pd.DataFrame({"column": mis.index, "n_missing": mis.values})
               .assign(pct=lambda d: (d["n_missing"] / len(df_train) * 100).round(2))
               .reset_index(drop=True))
print(missing_tbl)

print("\n=== Credit Score Distribution ===\n")
print(df_train['Credit_Score'].value_counts(normalize=True))

# Visuals - Credit Score Distribution Bar Graph

plt.figure(figsize=(8, 4))
sns.countplot(data=df_train, y='Credit_Score', color="#40AFFF")
plt.xticks(rotation=45, ha="right")
plt.ylabel("")
plt.title("Credit Score")
plt.tight_layout()
plt.show()

# 3) Cleaning Data
# ---------------------------

GARBAGE_TOKENS = ["_______", "_", "!@9#%8", "__10000__"]

# Numeric-looking columns that sometimes arrive as strings
NUMERIC_LIKE_COLS = [
    "Age", "Annual_Income", "Num_of_Loan", "Outstanding_Debt",
    "Num_of_Delayed_Payment", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Monthly_Inhand_Salary", "Total_EMI_per_month",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Amount_invested_monthly",
    "Monthly_Balance"
]

# Reasonable value ranges (outside -> NaN)
RANGE_RULES = {
    "Age": (1, 100),
    "Num_of_Loan": (0, 9),
    "Num_Credit_Card": (0, 11),
    "Interest_Rate": (0, 34),
    "Annual_Income": (0, 300000),
    "Total_EMI_per_month": (0, 5000),
    "Num_Bank_Accounts": (0, 100),
    "Num_of_Delayed_Payment": (0, 100),
}

def _coerce_numeric_strings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Keep digits, dot, minus; everything else stripped, then to numeric (NaN on failure)."""
    if not cols: 
        return df
    out = df.copy()
    pattern = re.compile(r"[^\d.\-]+")
    for c in cols:
        if c in out.columns:
            s = out[c].astype(str).str.replace(pattern, "", regex=True)
            out[c] = pd.to_numeric(s, errors="coerce")
    return out

def _apply_range_rules(df: pd.DataFrame, rules: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, (lo, hi) in rules.items():
        if col in out.columns:
            mask = out[col].between(lo, hi)
            out.loc[~mask, col] = np.nan
    return out

def _parse_credit_history_age_to_months(series: pd.Series) -> pd.Series:
    """
    'X Years Y Months' -> total months (float). If nothing parseable, returns NaN.
    No interpolation or filling here.
    """
    s = series.fillna("").astype(str).str.lower()
    years = pd.to_numeric(s.str.extract(r"(\d+)\s*year")[0], errors="coerce")
    months = pd.to_numeric(s.str.extract(r"(\d+)\s*month")[0], errors="coerce")

    # Fallback: first and second numbers seen
    fallback = s.str.findall(r"\d+")
    years = years.fillna(pd.to_numeric(fallback.str[0], errors="coerce"))
    months = months.fillna(pd.to_numeric(fallback.str[1], errors="coerce"))

    total = years.fillna(0) * 12 + months.fillna(0)
    total = total.replace(0, np.nan).astype(float)
    return total

def clean_credit_df_simple(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, opinionated cleaner for the credit dataset.
    - Drops PII columns if present
    - Normalizes obvious garbage tokens to NaN
    - Trims object strings; empty -> NaN
    - Coerces numeric-like string columns to float
    - Applies simple value-range guards (outside -> NaN)
    - Parses Credit_History_Age -> months (no interpolation), drops original
    - Converts selected text columns to 'category' (leaves NaNs as NaN)
    """
    df = df_raw.copy()

    # Drop Name and SSN
    df = df.drop(columns=[c for c in ["Name", "SSN"] if c in df.columns], errors="ignore")

    # Replace explicit junk tokens with NaN
    df = df.replace(GARBAGE_TOKENS, np.nan)

    # Normalize strings: strip whitespace; "" -> NaN
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].str.strip()
        df.loc[df[c] == "", c] = np.nan

    # Special casing without imputation: keep 'NM' as No
    if "Payment_of_Min_Amount" in df.columns:
        df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace({"NM": 'No'})

    # Coerce messy numerics
    present_numeric_like = [c for c in NUMERIC_LIKE_COLS if c in df.columns]
    df = _coerce_numeric_strings(df, present_numeric_like)

    # Drop out-of-range values to NaN
    df = _apply_range_rules(df, RANGE_RULES)

    # Credit_History_Age -> months (no interpolation), drop original text col
    if "Credit_History_Age" in df.columns:
        df["Credit_History_Age_Months"] = _parse_credit_history_age_to_months(df["Credit_History_Age"])
        df = df.drop(columns=["Credit_History_Age"])
    
    # Fills only where each customer's last non-null value exists for that column.
    if "Customer_ID" in df.columns:
        last_columns = [c for c in ["Age", "Occupation", "Type_of_Loan", "Credit_Mix"] if c in df.columns]

        def _last_non_null(s):
            s = s.dropna()
            return s.iloc[-1] if len(s) else np.nan

        df = df.sort_values(["Month"])
        temp_last = df.groupby("Customer_ID")[last_columns].agg(_last_non_null)

        for col in last_columns:
            df.loc[df[col].isna(), col] = df["Customer_ID"].map(temp_last[col])

        # Keep Age strictly numeric after the fill
        if "Age" in df.columns:
            df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Expand Type_of_Loan into multi-hot flags
    if "Type_of_Loan" in df.columns:
        # Normalize separators: turn "and" into commas; keep blanks as ""
        s = df["Type_of_Loan"].fillna("").astype(str)
        s = s.str.replace(r"\band\b", ",", regex=True)

        # Build 0/1 columns for each loan type
        dummies = s.str.get_dummies(sep=",")
        # Clean column names: strip & underscored
        dummies.columns = (dummies.columns
                           .str.strip()
                           .str.replace(r"\s+", "_", regex=True))
        # Drop any accidental empty-name column
        dummies = dummies.loc[:, dummies.columns.str.len() > 0]

        # Prefix to avoid collisions and cast to integers
        dummies = dummies.add_prefix("Loan_").astype("uint8")

        # Attach to frame and drop the original text column
        df = df.join(dummies)
        df = df.drop(columns=["Type_of_Loan"])

    return df


# Cleaning the training df
clean_train = clean_credit_df_simple(df_train)
mis_clean = clean_train.isna().sum()
mis_clean = mis_clean[mis_clean > 0].sort_values(ascending=False)
missing_tbl_clean = (pd.DataFrame({"column": mis_clean.index, "n_missing": mis_clean.values})
               .assign(pct=lambda d: (d["n_missing"] / len(df_train) * 100).round(2))
               .reset_index(drop=True))
print(missing_tbl_clean)

# Correlation Heatmap of original numeric features
num_cols = clean_train[[
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Credit_History_Age_Months"]]

r = num_cols.corr(numeric_only=True)

# Confusion matrix (normalized for readability)
capco_div = LinearSegmentedColormap.from_list(
    "capco_div",
    ["#005D9F", "#D9EFFF", "#005D9F"],  # dark ends, very light center
    N=256
)

ann = np.sign(r) * (r ** 2)
n  = len(r.columns)

fig, ax = plt.subplots(figsize=(min(14, 2 + 0.6*n), min(12, 2 + 0.6*n)))

sns.heatmap(
    r,                                   # colours reflect signed r
    cmap=capco_div,
    norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),  # 0 is lightest; ±1 darkest
    annot=ann, fmt=".2f",                # show signed r^2 with 2 decimals
    square=True,
    linewidths=1, linecolor="white",
    cbar_kws={"fraction": 0.046, "pad": 0.04},
    annot_kws={"size": max(8, 14 - n//3)}
)

ax.set_xticklabels(r.columns, rotation=45, ha="right")
ax.set_yticklabels(r.columns, rotation=0)
ax.set_title("Feature Correlation")
plt.tight_layout()
plt.show()

# Saving cleaned df
# clean_train.to_csv("clean_train.csv", index=False)


