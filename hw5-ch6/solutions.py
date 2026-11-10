"""Chapter 6, Problem 9: compare five regression methods on College."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "datasets" / "College.csv"
SEED = 0


def search(pipeline, parameters, X, y, folds):
    """Tune using training folds only, then refit on the entire training set."""
    fitted = GridSearchCV(
        pipeline,
        parameters,
        scoring="neg_mean_squared_error",
        cv=folds,
        n_jobs=1,
        refit=True,
        error_score="raise",
    ).fit(X, y)
    rows = []
    for i, params in enumerate(fitted.cv_results_["params"]):
        values = {
            k: int(v) if isinstance(v, (int, np.integer)) else float(v)
            for k, v in params.items()
        }
        rows.append({
            "parameters": values,
            "mean_validation_mse": float(-fitted.cv_results_["mean_test_score"][i]),
            "fold_validation_mse": [
                float(-fitted.cv_results_[f"split{j}_test_score"][i])
                for j in range(len(folds))
            ],
        })
    assert fitted.best_index_ == int(np.argmin([
        row["mean_validation_mse"] for row in rows
    ]))
    return fitted, rows


def main():
    warnings.simplefilter("error", ConvergenceWarning)
    college = pd.read_csv(DATA_PATH, index_col=0)
    assert college.index.is_unique
    assert not college.isna().any().any()
    assert set(college["Private"].unique()) == {"Yes", "No"}
    X = college.drop(columns="Apps").copy()
    X["Private"] = (X["Private"] == "Yes").astype(int)
    X = X.astype(float)
    y = college["Apps"].to_numpy(dtype=float)
    assert X.shape == (777, 17)
    assert np.isfinite(X.to_numpy()).all() and np.isfinite(y).all()

    train, test = train_test_split(
        np.arange(len(college)), test_size=0.30, random_state=SEED
    )
    assert len(set(train) & set(test)) == 0
    assert sorted(np.r_[train, test]) == list(range(len(college)))
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y[train], y[test]
    folds = list(KFold(n_splits=10, shuffle=True, random_state=SEED).split(X_train))
    validation_positions = np.concatenate([valid for _, valid in folds])
    assert sorted(validation_positions) == list(range(len(train)))
    assert all(not set(fit) & set(valid) for fit, valid in folds)

    def pipeline(estimator):
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])

    models = {}
    tuning = {}
    searches = {}
    models["Least squares"] = pipeline(LinearRegression()).fit(X_train, y_train)

    specifications = [
        ("Ridge", pipeline(Ridge()),
         {"model__alpha": np.logspace(-4, 8, 121)}),
        ("Lasso", pipeline(Lasso(max_iter=200000, tol=1e-10)),
         {"model__alpha": np.logspace(-2, 4, 121)}),
        ("PCR", Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(svd_solver="full")),
            ("model", LinearRegression()),
         ]), {"pca__n_components": list(range(1, X.shape[1] + 1))}),
        ("PLS", pipeline(PLSRegression(scale=False, max_iter=5000, tol=1e-10)),
         {"model__n_components": list(range(1, X.shape[1] + 1))}),
    ]
    for name, estimator, parameters in specifications:
        print(f"Cross-validating {name}...", flush=True)
        fitted, rows = search(estimator, parameters, X_train, y_train, folds)
        models[name] = fitted.best_estimator_
        searches[name] = fitted
        tuning[name] = rows
        if name in {"Ridge", "Lasso"}:
            assert 0 < fitted.best_index_ < len(rows) - 1, (
                f"Expand the penalty grid for {name}: optimum is at an endpoint."
            )

    results = {}
    for name, fitted in models.items():
        predicted = np.asarray(fitted.predict(X_test)).reshape(-1)
        assert predicted.shape == y_test.shape and np.isfinite(predicted).all()
        mse = float(mean_squared_error(y_test, predicted))
        assert np.isclose(mse, np.sum((y_test - predicted) ** 2) / len(test))
        result = {
            "test_mse": mse,
            "test_rmse": float(np.sqrt(mse)),
            "test_r2": float(r2_score(y_test, predicted)),
            "test_predictions": predicted.tolist(),
        }
        if name in searches:
            cv = searches[name]
            result["cv_mse"] = float(-cv.best_score_)
            result["selected_parameters"] = {
                key: int(value) if isinstance(value, (int, np.integer)) else float(value)
                for key, value in cv.best_params_.items()
            }
        if name in {"Ridge", "Lasso"}:
            result["lambda"] = float(fitted.named_steps["model"].alpha)
        if name in {"PCR", "PLS"}:
            component_step = "pca" if name == "PCR" else "model"
            result["components"] = int(fitted.named_steps[component_step].n_components)
        if name == "Lasso":
            coefficients = fitted.named_steps["model"].coef_
            result["nonzero_predictors"] = int(np.count_nonzero(coefficients))
            result["retained_predictors"] = X.columns[coefficients != 0].tolist()
            result["standardized_coefficients"] = dict(zip(
                X.columns, map(float, coefficients)
            ))
        results[name] = result
        print(name, {k: v for k, v in result.items() if k in {
            "test_mse", "test_rmse", "test_r2", "lambda", "components",
            "nonzero_predictors", "cv_mse",
        }}, flush=True)

    # Independent algebraic checks of fitting and metric calculations.
    scaler = models["Least squares"].named_steps["scale"]
    Z_train, Z_test = scaler.transform(X_train), scaler.transform(X_test)
    A_train = np.column_stack([np.ones(len(train)), Z_train])
    A_test = np.column_stack([np.ones(len(test)), Z_test])
    assert np.linalg.matrix_rank(A_train) == 18
    beta = np.linalg.lstsq(A_train, y_train, rcond=None)[0]
    ols_prediction = np.asarray(results["Least squares"]["test_predictions"])
    np.testing.assert_allclose(A_test @ beta, ols_prediction, rtol=1e-9, atol=1e-7)

    ridge_lambda = results["Ridge"]["lambda"]
    ridge_beta = np.linalg.solve(
        Z_train.T @ Z_train + ridge_lambda * np.eye(X.shape[1]),
        Z_train.T @ (y_train - y_train.mean()),
    )
    np.testing.assert_allclose(
        y_train.mean() + Z_test @ ridge_beta,
        results["Ridge"]["test_predictions"], rtol=1e-9, atol=1e-7,
    )

    lasso = models["Lasso"]
    lasso_beta = lasso.named_steps["model"].coef_
    residual = y_train - lasso.predict(X_train)
    correlations = Z_train.T @ residual / len(train)
    active = lasso_beta != 0
    lam = results["Lasso"]["lambda"]
    active_error = float(np.max(np.abs(
        correlations[active] - lam * np.sign(lasso_beta[active])
    ))) if active.any() else 0.0
    inactive_excess = float(np.max(np.maximum(
        np.abs(correlations[~active]) - lam, 0
    ))) if (~active).any() else 0.0
    assert active_error < 1e-4 and inactive_excess < 1e-4

    all_components_pcr = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=X.shape[1], svd_solver="full")),
        ("model", LinearRegression()),
    ]).fit(X_train, y_train)
    np.testing.assert_allclose(
        all_components_pcr.predict(X_test), ols_prediction, rtol=1e-9, atol=1e-7
    )
    all_components_pls = pipeline(PLSRegression(
        n_components=X.shape[1], scale=False, max_iter=5000, tol=1e-10
    )).fit(X_train, y_train)
    np.testing.assert_allclose(
        np.asarray(all_components_pls.predict(X_test)).reshape(-1),
        ols_prediction, rtol=1e-8, atol=1e-5,
    )

    baseline_prediction = np.repeat(y_train.mean(), len(test))
    baseline_mse = float(mean_squared_error(y_test, baseline_prediction))
    output = {
        "data": {
            "source": "../datasets/College.csv",
            "sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            "n": len(college), "p": X.shape[1],
            "predictors": X.columns.tolist(), "response": "Apps",
            "private_encoding": {"Yes": 1, "No": 0},
        },
        "split": {
            "seed": SEED, "test_fraction": 0.3,
            "n_train": len(train), "n_test": len(test),
            "train_positions": train.tolist(), "test_positions": test.tolist(),
        },
        "cross_validation": {
            "folds": 10, "shuffle": True, "seed": SEED,
            "selection": "minimum mean validation MSE",
            "preprocessing": "scaling and PCA fitted within each training fold",
            "validation_positions_in_training_set": [valid.tolist() for _, valid in folds],
            "search_results": tuning,
        },
        "models": results,
        "baseline": {
            "description": "predict the training-set mean for every test observation",
            "test_mse": baseline_mse, "test_rmse": float(np.sqrt(baseline_mse)),
        },
        "test_actual": y_test.tolist(),
        "checks": {
            "disjoint_train_test": True, "all_observations_used_once": True,
            "cross_validation_partition": True, "finite_predictions": True,
            "manual_mse": True, "ols_matches_lstsq": True,
            "ridge_matches_closed_form": True,
            "lasso_kkt_active_max_error": active_error,
            "lasso_kkt_inactive_max_excess": inactive_excess,
            "full_pcr_matches_ols": True, "full_pls_matches_ols": True,
            "penalties_not_grid_endpoints": True,
        },
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        },
    }
    (HERE / "results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print("All numerical checks passed; wrote results.json.", flush=True)


if __name__ == "__main__":
    main()
