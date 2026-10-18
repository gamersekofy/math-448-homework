# ruff: noqa
"""
MATH 448: Statistical Learning — Homework 3 (Chapter 4: Classification)
Problems 3, 6, 13, 15

Author:  Uzair Hamed Mohammed
Date:    10/16/2026

This script produces every numerical result and figure used in the solutions PDF.
Run:  python solutions.py
Outputs:  results.json  (all numeric tables / values)
          figs/*.png    (all figures)
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from ISLP import load_data
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

RNG = np.random.default_rng(0)  # single seeded generator for any randomness

plt.rcParams.update(
    {
        "figure.dpi": 125,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10.5,
        "axes.titlesize": 11.5,
    }
)

# All numeric results accumulate into this dict, dumped to results.json at the end.
RESULTS: dict = {}


def savefig(fig: plt.Figure, name: str) -> None:
    path = os.path.join(FIGS, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}")


# =====================================================================
# Question 3 — QDA with p = 1 is quadratic, not linear
# =====================================================================
# Written proof goes in the PDF.  Here we illustrate it numerically:
# the Bayes boundary solves delta_1(x) = delta_2(x), which is quadratic
# whenever sigma_1^2 != sigma_2^2.  Equal variances give the single linear
# boundary (left panel); unequal variances give two roots (right panel).

MU1, S1SQ = -1.5, 1.0  # class 1: mu_1, sigma_1^2
MU2, S2SQ = 1.5, 4.0  # class 2: mu_2, sigma_2^2

xgrid = np.linspace(-6, 6, 1200)


def delta(x: np.ndarray, mu: float, var: float, logprior: float = 0.0) -> np.ndarray:
    """Log-posterior delta_k(x) for a N(mu, var) class density (p = 1)."""
    return logprior - 0.5 * np.log(var) - (x - mu) ** 2 / (2 * var)


def del_k_minus_del_l_squared_terms(
    mu1: float, v1: float, mu2: float, v2: float
) -> tuple[float, float, float]:
    """Coefficients (a, b, c) of the x^2, x, const terms of delta_1 - delta_2.

    delta_1 - delta_2 = -log(sigma_1/sigma_2) - 0.5 [ (x-mu_1)^2/v_1 - (x-mu_2)^2/v_2 ]
    After multiplying through by -1 (the equation delta_1 = delta_2 == a x^2 + b x + c = 0 with log terms folded into c):
      a = 0.5 (1/v1 - 1/v2)
      b = -(mu1/v1 - mu2/v2)
      c = 0.5 (mu1^2/v1 - mu2^2/v2)
    plus the prior/normalization constants, returned separately.
    """
    a = 0.5 * (1 / v1 - 1 / v2)
    b = -(mu1 / v1 - mu2 / v2)
    c = 0.5 * (mu1**2 / v1 - mu2**2 / v2)
    return a, b, c


def boundary_roots(
    mu1: float, v1: float, mu2: float, v2: float, log_prior_ratio: float = 0.0
) -> np.ndarray:
    """Roots of delta_1(x) = delta_2(x), equiprobable classes by default.

    Solves  a x^2 + b x + c - log(pi1/pi2) - (1/2) log(v2/v1) = 0,
    i.e. includes the variance-normalization term of the log posterior.
    """
    a, b, c = del_k_minus_del_l_squared_terms(mu1, v1, mu2, v2)
    c = c - log_prior_ratio - 0.5 * np.log(v2 / v1)
    return np.roots([a, b, c]) if abs(a) > 1e-12 else np.array([-c / b])


eq_roots = boundary_roots(MU1, 1.0, MU2, 1.0)
uneq_roots = boundary_roots(
    MU1, S1SQ, MU2, S2SQ, log_prior_ratio=float(np.log(0.5) - np.log(0.5))
)
_a, _b, _c = del_k_minus_del_l_squared_terms(MU1, S1SQ, MU2, S2SQ)
_c_full = _c - 0.5 * np.log(S2SQ / S1SQ)  # equal priors, variance-norm term kept
RESULTS["q3"] = {
    "equal_var_roots": [float(r) for r in eq_roots],
    "unequal_var_roots": [float(r) for r in uneq_roots],
    "a_quad": float(_a),
    "b_quad": float(_b),
    "c_quad_full": float(_c_full),
}

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

# --- Panel A: equal variances -> linear boundary ---------------------
for mu, var, c, lab in [
    (MU1, 1.0, "#1f77b4", r"$N(\mu_1,\ \sigma^2)$"),
    (MU2, 1.0, "#d62728", r"$N(\mu_2,\ \sigma^2)$"),
]:
    d = delta(xgrid, mu, var, np.log(0.5))
    axes[0].plot(xgrid, d, color=c, label=lab)
    axes[0].fill_between(xgrid, -8, d, color=c, alpha=0.08)
for r in eq_roots:
    axes[0].axvline(r, color="k", ls="--", lw=1.2)
axes[0].set_title("Equal variances: one linear boundary")
axes[0].set_xlabel("$x$")
axes[0].set_ylabel(r"$\delta_k(x)$")
axes[0].set_ylim(-8, 0.4)
axes[0].legend(loc="lower center", fontsize=9)

# --- Panel B: unequal variances -> quadratic boundary ----------------
for mu, var, c, lab in [
    (MU1, S1SQ, "#1f77b4", rf"$N(\mu_1={MU1:g},\ \sigma_1^2={S1SQ:g})$"),
    (MU2, S2SQ, "#d62728", rf"$N(\mu_2={MU2:g},\ \sigma_2^2={S2SQ:g})$"),
]:
    d = delta(xgrid, mu, var, np.log(0.5))
    axes[1].plot(xgrid, d, color=c, label=lab)
    axes[1].fill_between(xgrid, -8, d, color=c, alpha=0.08)
for r in uneq_roots:
    axes[1].axvline(r, color="k", ls="--", lw=1.2)
axes[1].set_title("Unequal variances: two quadratic boundaries")
axes[1].set_xlabel("$x$")
axes[1].set_ylabel(r"$\delta_k(x)$")
axes[1].set_ylim(-8, 0.4)
axes[1].legend(loc="lower center", fontsize=9)

fig.tight_layout()
savefig(fig, "q3_qda_boundary.png")

# =====================================================================
# Question 6 — logistic regression probability estimates
# =====================================================================
# beta0 = -6, beta1 = 0.05 (hours), beta2 = 1 (GPA)   [Y = receive an A]

B0, B1, B2 = -6.0, 0.05, 1.0


def logit_to_prob(eta: float) -> float:
    return float(1 / (1 + np.exp(-eta)))


q6a = logit_to_prob(B0 + B1 * 40 + B2 * 3.5)
# 50% chance <=> logit = 0 <=> -6 + 0.05 h + 3.5 = 0  <=>  h = 50
q6b = (6.0 - B2 * 3.5) / B1
q6b_check = logit_to_prob(B0 + B1 * q6b + B2 * 3.5)
RESULTS["q6"] = {
    "eta_a": float(B0 + B1 * 40 + B2 * 3.5),
    "p_a": q6a,
    "hours_b": float(q6b),
    "p_b_check": q6b_check,
}
print(f"Q6: (a) p = {q6a:.4f};  (b) h = {q6b:.2f} h")


# =====================================================================
# Question 13 — the Weekly data set
# =====================================================================
Weekly = load_data("Weekly")
RESULTS["q13_shape"] = list(Weekly.shape)

# ---------------------------------------------------------------- (a)
print("Q13(a): summaries")
desc = Weekly.describe()
RESULTS["q13a_describe"] = {
    c: {k: float(v) for k, v in desc[c].items()} for c in desc.columns
}
RESULTS["q13a_direction_counts"] = {
    k: int(v) for k, v in Weekly["Direction"].value_counts().items()
}
RESULTS["q13a_years"] = {
    "min": int(Weekly["Year"].min()),
    "max": int(Weekly["Year"].max()),
}

# Correlations between the numeric predictors
corr = Weekly.select_dtypes("number").corr()
RESULTS["q13a_corr"] = {
    c: {k: float(v) for k, v in corr[c].items()} for c in corr.columns
}
gui = Weekly[["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume"]]

# --- Figure: boxplots of Lags (and Today) by Direction ---------------
fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
for ax, col in zip(axes.ravel(), ["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Today"]):
    sns.boxplot(
        data=Weekly, x="Direction", y=col, hue="Direction", ax=ax, legend=False,
        palette=["#c44e52", "#4c72b0"], order=["Down", "Up"], width=0.55,
        fliersize=2.5,
    )
    ax.set_xlabel("")
    ax.set_title(f"{col} by Direction")
fig.tight_layout()
savefig(fig, "q13a_boxplots.png")

# --- Figure: volume growth over time ---------------------------------
fig, ax = plt.subplots(figsize=(9.2, 3.4))
ax.plot(np.arange(len(Weekly)), Weekly["Volume"], lw=0.7, color="#9fb3c8", alpha=0.9)
yearly = Weekly.groupby("Year")["Volume"].mean()
yearly_x = (yearly.index - 1990) * 52 + 26
ax.plot(yearly_x, yearly.values, color="#c44e52", lw=2.2, label="Yearly mean volume")
ax.set_xlabel("Year")
ax.set_ylabel("Volume")
ax.set_title("Weekly share volume grows over time")
ax.set_xticks(np.arange(0, len(Weekly) + 1, 104))
ax.set_xticklabels([str(1990 + w // 52) for w in np.arange(0, len(Weekly) + 1, 104)])
ax.legend()
fig.tight_layout()
savefig(fig, "q13a_volume.png")

# --- Figure: correlation heatmap --------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 5.4))
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
    square=True, cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 8.5},
)
ax.set_title("Correlations among numeric Weekly variables")
ax.grid(False)
fig.tight_layout()
savefig(fig, "q13a_corr.png")

# ---------------------------------------------------------------- (b)
print("Q13(b): logistic regression, full data, 5 lags + Volume")
preds_full = ["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume"]
y_full = (Weekly["Direction"] == "Up").astype(int)
X_full = sm.add_constant(Weekly[preds_full], has_constant="add")
glm_full = sm.GLM(y_full, X_full, family=sm.families.Binomial()).fit()

tbl = glm_full.summary2().tables[1]
RESULTS["q13b_smry"] = {
    coef: {
        "coef": float(row["Coef."]),
        "se": float(row["Std.Err."]),
        "z": float(row["z"]),
        "p": float(row["P>|z|"]),
    }
    for coef, row in tbl.iterrows()
}
print(tbl.round(4).to_string())

# ---------------------------------------------------------------- (c)
yhat_prob_full = glm_full.predict(X_full)
yhat_full = np.where(yhat_prob_full > 0.5, "Up", "Down")
act_full = Weekly["Direction"].astype(str).to_numpy()

RESULTS["q13c_confusion"] = {
    "matrix": confusion_matrix(act_full, yhat_full, labels=["Down", "Up"]).tolist(),
    "labels": ["Down", "Up"],
    "n": int(len(act_full)),
    "correct": int((act_full == yhat_full).sum()),
    "frac": float((act_full == yhat_full).mean()),
    "down_correct_rate": float((yhat_full[act_full == "Down"] == "Down").mean()),
    "up_correct_rate": float((yhat_full[act_full == "Up"] == "Up").mean()),
}
print(f"  training fraction correct = {RESULTS['q13c_confusion']['frac']:.4f}")


# ---------- helpers for the (d)-(j) train/test experiments ------------
train_mask = Weekly["Year"] <= 2008
test_mask = Weekly["Year"] >= 2009


def fit_summary(
    method: str,
    predictors: list[str],
    Xtr: pd.DataFrame,
    ytr_bin: np.ndarray,
    ytr_lab: np.ndarray,
    Xte: pd.DataFrame,
    yte_lab: np.ndarray,
    k: int | None = None,
) -> dict:
    """Fit one of the five methods, return accuracy + confusion matrix."""
    if method == "Logit":
        fitG = sm.GLM(
            ytr_bin, sm.add_constant(Xtr, has_constant="add"),
            family=sm.families.Binomial(),
        ).fit()
        pr = fitG.predict(sm.add_constant(Xte, has_constant="add"))
        yh = np.where(pr > 0.5, "Up", "Down")
    elif method == "LDA":
        yh = LinearDiscriminantAnalysis().fit(Xtr, ytr_lab).predict(Xte)
    elif method == "QDA":
        yh = QuadraticDiscriminantAnalysis().fit(Xtr, ytr_lab).predict(Xte)
    elif method == "NB":
        yh = GaussianNB().fit(Xtr, ytr_lab).predict(Xte)
    elif method == "KNN":
        pipe = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=k if k is not None else 1),
        ).fit(Xtr, ytr_lab)
        yh = pipe.predict(Xte)
    else:
        raise ValueError(method)
    cm = confusion_matrix(yte_lab, yh, labels=["Down", "Up"])
    out = {
        "method": method,
        "predictors": list(predictors),
        "matrix": cm.tolist(),
        "correct": int((yh == yte_lab).sum()),
        "n": int(len(yte_lab)),
        "frac": float((yh == yte_lab).mean()),
    }
    if k is not None:
        out["k"] = int(k)
    return out


# --------------------------------------------------- (d) through (h)
print("Q13(d-h): Lag2 only, train 1990-2008 / test 2009-2010")
lag2 = ["Lag2"]
Xtr_df = Weekly.loc[train_mask, lag2]
Xte_df = Weekly.loc[test_mask, lag2]
ytr_lab = Weekly.loc[train_mask, "Direction"].astype(str).to_numpy()
yte_lab = Weekly.loc[test_mask, "Direction"].astype(str).to_numpy()
ytr_bin = (Weekly.loc[train_mask, "Direction"] == "Up").astype(int).to_numpy()

results_lag2 = {
    m: fit_summary(m, lag2, Xtr_df, ytr_bin, ytr_lab, Xte_df, yte_lab, k=1 if m == "KNN" else None)
    for m in ["Logit", "LDA", "QDA", "KNN", "NB"]
}
RESULTS["q13dh_lag2"] = results_lag2
for m, r in results_lag2.items():
    print(f"  {m:5s}: {r['correct']:3d}/{r['n']} correct = {r['frac']:.4f}")

# ------------------------------------------------------------- (j)
print("Q13(j): systematic experiment")
W = Weekly.copy()
W["Lag2_sq"] = W["Lag2"] ** 2
W["Lag1_sq"] = W["Lag1"] ** 2
W["Lag1Lag2"] = W["Lag1"] * W["Lag2"]
W["logVol"] = np.log(W["Volume"])

sets: list[list[str]] = [
    ["Lag2"],
    ["Lag1", "Lag2"],
    ["Lag2", "Lag3"],
    ["Lag1", "Lag2", "Lag3"],
    ["Lag2", "Lag5"],
    ["Lag1", "Lag2", "Lag3", "Lag4", "Lag5"],
    ["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume"],
    ["Lag2", "Volume"],
    ["Lag2", "Lag2_sq"],
    ["Lag2", "Lag2_sq", "Lag1Lag2"],
    ["Lag1", "Lag2", "Lag1Lag2"],
    ["Lag1", "Lag2", "Lag1_sq", "Lag2_sq"],
    ["Lag1", "Lag2", "Volume"],
    ["Lag2", "logVol"],
    ["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume", "Lag1Lag2"],
]
METHODS = ["Logit", "LDA", "QDA", "NB"]
KNNS = [1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30, 50]

exp_rows: list[dict] = []
for s in sets:
    Xtr = W.loc[train_mask, s]
    Xte = W.loc[test_mask, s]
    ytr_b = (W.loc[train_mask, "Direction"] == "Up").astype(int).to_numpy()
    ytr_l = W.loc[train_mask, "Direction"].astype(str).to_numpy()
    for m in METHODS:
        exp_rows.append(fit_summary(m, s, Xtr, ytr_b, ytr_l, Xte, yte_lab))
    for k in KNNS:
        exp_rows.append(fit_summary("KNN", s, Xtr, ytr_b, ytr_l, Xte, yte_lab, k=k))
    print(f"  done: {s}")

RESULTS["q13j_all"] = exp_rows
best = max(exp_rows, key=lambda r: r["frac"])
RESULTS["q13j_best"] = best
print(f"  BEST: {best['method']} {best['predictors']}"
      + (f" (K={best.get('k')})" if "k" in best else "")
      + f" -> {best['frac']:.4f}")

rank_table = []
for s in sets:
    row: dict = {"predictors": "+".join(s)}
    for m in METHODS:
        acc = next(r["frac"] for r in exp_rows if r["method"] == m and r["predictors"] == s)
        row[m] = acc
    knn_r = [r for r in exp_rows if r["method"] == "KNN" and r["predictors"] == s]
    best_knn = max(knn_r, key=lambda r: r["frac"])
    row["KNN"] = best_knn["frac"]
    row["K*"] = best_knn["k"]
    rank_table.append(row)
rank_table.sort(key=lambda r: -max(r[m] for m in METHODS + ["KNN"]))
RESULTS["q13j_rank_table"] = rank_table

# KNN K-sweep on Lag2 for the plot
K_VALUE = 1
knn_curve = []
for k in range(1, 51):
    r = fit_summary("KNN", lag2, Xtr_df, ytr_bin, ytr_lab, Xte_df, yte_lab, k=k)
    knn_curve.append({"k": k, "frac": r["frac"]})
RESULTS["q13j_knn_curve_lag2"] = knn_curve

fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot([r["k"] for r in knn_curve], [r["frac"] for r in knn_curve],
        marker="o", ms=3.5, lw=1.2, color="#4c72b0")
ax.set_xlabel("K (number of neighbors)")
ax.set_ylabel("Test accuracy")
ax.set_title("KNN on Lag2: test accuracy vs. K")
fig.tight_layout()
savefig(fig, "q13j_knn_sweep.png")

# =====================================================================
# Question 15 — Power, Power2, Power3, PlotPower
# =====================================================================
print("Q15: power functions")


# (a)
def Power() -> None:
    """Print 2**3."""
    print(2**3)


# (b)
def Power2(x, a) -> None:
    """Print x**a."""
    print(x**a)


# (d)
def Power3(x, a):
    """Return x**a."""
    result = x**a
    return result


# (f)
def PlotPower(x: np.ndarray, a) -> plt.Figure:
    """Plot x**a over the vector x."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, Power3(x, a), marker="o", color="#4c72b0", lw=1.8, ms=4)
    ax.set_xlabel("x")
    if isinstance(a, int):
        ax.set_ylabel(f"$x^{a}$")
    else:
        ax.set_ylabel(r"$x^a$")
    ax.set_title(rf"$y = x^{a}$" if isinstance(a, int) else r"$y = x^a$")
    fig.tight_layout()
    return fig


print("(a) Power() output:")
Power()

RESULTS["q15"] = {
    "Power2_3_8": None,  # filled below (Power2 only prints)
    "c": {
        "10**3": int(Power3(10, 3)),
        "8**17": int(Power3(8, 17)),
        "131**3": int(Power3(131, 3)),
    },
}

print("(b) Power2(3, 8) output:")
Power2(3, 8)
RESULTS["q15"]["Power2_3_8"] = int(3**8)

# (e) plot f(x) = x^2 on 1..10, linear + log-log panels
x10 = np.arange(1, 11)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
axes[0].plot(x10, Power3(x10, 2), marker="o", color="#4c72b0", lw=1.8, ms=4)
axes[0].set_xlabel("$x$")
axes[0].set_ylabel("$x^2$")
axes[0].set_title(r"$f(x) = x^2$ (linear scale)")
axes[1].plot(x10, Power3(x10, 2), marker="o", color="#c44e52", lw=1.8, ms=4)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel(r"$x$ (log scale)")
axes[1].set_ylabel(r"$x^2$ (log scale)")
axes[1].set_title(r"$f(x) = x^2$ (log-log scale)")
fig.tight_layout()
savefig(fig, "q15_power3.png")

# (f) PlotPower(np.arange(1, 11), 3)
fig = PlotPower(np.arange(1, 11), 3)
savefig(fig, "q15_plotpower.png")

# =====================================================================
# dump results
# =====================================================================
with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(RESULTS, fh, indent=1, default=str)
print("wrote results.json")
