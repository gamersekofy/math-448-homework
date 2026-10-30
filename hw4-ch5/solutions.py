# ruff: noqa
"""MATH 448 — Homework 4 (Chapter 5: Resampling Methods).

Problems 3, 5, 6, 7.  This script computes every numerical result used in
solutions.typ and writes them to results.json.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP import load_data
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
RESULTS: dict = {}


def design(data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Design matrix with an intercept, as a plain float array."""
    return sm.add_constant(
        np.asarray(data[columns], dtype=float), has_constant="add"
    )


def fit_logit(data: pd.DataFrame, y: pd.Series, columns: list[str]):
    """Binomial GLM fit with an intercept."""
    return sm.GLM(
        np.asarray(y, dtype=float),
        design(data, columns),
        family=sm.families.Binomial(),
    ).fit()


def coef_table(fit, columns: list[str]) -> dict:
    """Collect coefficient, standard error, z statistic, and p-value."""
    names = ["const", *columns]
    return {
        name: {
            "coef": float(fit.params[j]),
            "se": float(fit.bse[j]),
            "z": float(fit.tvalues[j]),
            "p": float(fit.pvalues[j]),
        }
        for j, name in enumerate(names)
    }


def validation_error(
    data: pd.DataFrame, y: pd.Series, columns: list[str], seed: int
) -> dict:
    """One 50/50 validation-set split and the resulting validation error."""
    n = len(data)
    train_idx, valid_idx = train_test_split(
        np.arange(n), test_size=0.5, random_state=seed, shuffle=True
    )
    fit = fit_logit(data.iloc[train_idx], y.iloc[train_idx], columns)
    prob = fit.predict(design(data.iloc[valid_idx], columns))
    pred = (prob > 0.5).astype(int)
    actual = np.asarray(y.iloc[valid_idx], dtype=int)
    cm = confusion_matrix(actual, pred, labels=[0, 1])
    return {
        "seed": int(seed),
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "error": float(np.mean(pred != actual)),
        "n_errors": int(np.sum(pred != actual)),
        "confusion": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


# =====================================================================
# Problem 5 — validation-set estimate on Default
# =====================================================================
Default = load_data("Default").copy()
Default["student01"] = (Default["student"] == "Yes").astype(int)
y_default = (Default["default"] == "Yes").astype(int)

base_cols = ["income", "balance"]
student_cols = ["income", "balance", "student01"]

fit_base = fit_logit(Default, y_default, base_cols)
fit_student = fit_logit(Default, y_default, student_cols)

RESULTS["q5a"] = {
    "n": int(len(Default)),
    "n_default": int(y_default.sum()),
    "coef": coef_table(fit_base, base_cols),
}
RESULTS["q5d_full_model"] = {"coef": coef_table(fit_student, student_cols)}

base_splits = [validation_error(Default, y_default, base_cols, SEED + i) for i in range(3)]
student_splits = [
    validation_error(Default, y_default, student_cols, SEED + i) for i in range(3)
]

RESULTS["q5b"] = base_splits[0]
RESULTS["q5c"] = {
    "splits": base_splits,
    "mean_error": float(np.mean([r["error"] for r in base_splits])),
    "sd_error": float(np.std([r["error"] for r in base_splits], ddof=1)),
}
RESULTS["q5d"] = {
    "splits": student_splits,
    "mean_error": float(np.mean([r["error"] for r in student_splits])),
    "differences": [
        float(s["error"] - b["error"]) for s, b in zip(student_splits, base_splits)
    ],
}

# =====================================================================
# Problem 6 — bootstrap standard errors
# =====================================================================


def boot_fn(data: pd.DataFrame, index: np.ndarray) -> dict:
    """Coefficient estimates for income and balance on a bootstrap sample."""
    sample = data.iloc[index]
    y_sample = (sample["default"] == "Yes").astype(int)
    fit = fit_logit(sample, y_sample, base_cols)
    return {"income": float(fit.params[1]), "balance": float(fit.params[2])}


RESULTS["q6a"] = {
    "coef": coef_table(fit_base, base_cols),
    "se_income": float(fit_base.bse[1]),
    "se_balance": float(fit_base.bse[2]),
}

rng = np.random.default_rng(SEED)
B = 1000
n = len(Default)
boot = [boot_fn(Default, rng.integers(0, n, n)) for _ in range(B)]
boot_df = pd.DataFrame(boot)
RESULTS["q6c"] = {
    "B": B,
    "mean_income": float(boot_df["income"].mean()),
    "mean_balance": float(boot_df["balance"].mean()),
    "se_income": float(boot_df["income"].std(ddof=1)),
    "se_balance": float(boot_df["balance"].std(ddof=1)),
}

# =====================================================================
# Problem 7 — manual LOOCV on Weekly
# =====================================================================
Weekly = load_data("Weekly").copy()
y_weekly = (Weekly["Direction"] == "Up").astype(int)
lag_cols = ["Lag1", "Lag2"]
n_weekly = len(Weekly)

fit_weekly = fit_logit(Weekly, y_weekly, lag_cols)
RESULTS["q7a"] = {"coef": coef_table(fit_weekly, lag_cols), "n": int(n_weekly)}

mask_first = np.arange(1, n_weekly)
fit_wo_first = fit_logit(Weekly.iloc[mask_first], y_weekly.iloc[mask_first], lag_cols)
p_first = float(fit_wo_first.predict(design(Weekly.iloc[[0]], lag_cols))[0])
pred_first = "Up" if p_first > 0.5 else "Down"
actual_first = str(Weekly["Direction"].iloc[0])
RESULTS["q7b"] = {"coef": coef_table(fit_wo_first, lag_cols)}
RESULTS["q7c"] = {
    "p_up": p_first,
    "predicted": pred_first,
    "actual": actual_first,
    "correct": bool(pred_first == actual_first),
}

errors: list[int] = []
for i in range(n_weekly):
    keep = np.ones(n_weekly, dtype=bool)
    keep[i] = False
    idx = np.flatnonzero(keep)
    fit_i = fit_logit(Weekly.iloc[idx], y_weekly.iloc[idx], lag_cols)
    p_i = float(fit_i.predict(design(Weekly.iloc[[i]], lag_cols))[0])
    pred_i = 1 if p_i > 0.5 else 0
    actual_i = int(y_weekly.iloc[i])
    errors.append(int(pred_i != actual_i))

errors_arr = np.asarray(errors)
RESULTS["q7d"] = {"errors": errors_arr.tolist(), "n_errors": int(errors_arr.sum())}
RESULTS["q7e"] = {
    "n": int(n_weekly),
    "n_errors": int(errors_arr.sum()),
    "loocv_error": float(errors_arr.mean()),
    "loocv_accuracy": float(1.0 - errors_arr.mean()),
    "majority_baseline": float(max(y_weekly.mean(), 1.0 - y_weekly.mean())),
}

with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(RESULTS, fh, indent=1)

print("Q5(a) full model:")
print(pd.DataFrame(RESULTS["q5a"]["coef"]).T.round(6).to_string())
print("\nQ5 validation errors (income + balance):")
for r in base_splits:
    print(f"  seed {r['seed']}: {r['n_errors']}/{r['n_valid']} = {r['error']:.4f}")
print("\nQ5 validation errors (income + balance + student):")
for r in student_splits:
    print(f"  seed {r['seed']}: {r['n_errors']}/{r['n_valid']} = {r['error']:.4f}")
print("\nQ6 GLM and bootstrap standard errors:")
print(f"  income : GLM {RESULTS['q6a']['se_income']:.9f}  bootstrap {RESULTS['q6c']['se_income']:.9f}")
print(f"  balance: GLM {RESULTS['q6a']['se_balance']:.9f}  bootstrap {RESULTS['q6c']['se_balance']:.9f}")
print("\nQ7(c) first observation:")
print(
    f"  P(Up) = {p_first:.4f}, predicted {pred_first}, actual {actual_first}, "
    f"correct = {RESULTS['q7c']['correct']}"
)
print(
    f"Q7(e) LOOCV: {RESULTS['q7e']['n_errors']}/{n_weekly} = "
    f"{RESULTS['q7e']['loocv_error']:.5f}"
)
print("wrote results.json")
