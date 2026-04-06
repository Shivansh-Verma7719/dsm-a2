import argparse
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.panel import Panel
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "report" / "p2" / "data" / "predictive_modelling"
FIG_ROOT = ROOT / "report" / "p2" / "figures" / "predictive_modelling"
console = Console()


class HurdleUsefulRegressor(BaseEstimator, RegressorMixin):
    """Two-stage model: P(useful>0) * E[useful | useful>0]."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.classifier: HistGradientBoostingClassifier | None = None
        self.regressor: HistGradientBoostingRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_clean = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0)
        y_bin = (y_clean > 0).astype(int)

        self.classifier = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=220,
            l2_regularization=0.02,
            random_state=self.random_state,
        )
        self.classifier.fit(X, y_bin)

        pos_mask = y_bin == 1
        if int(pos_mask.sum()) == 0:
            self.regressor = HistGradientBoostingRegressor(random_state=self.random_state)
            self.regressor.fit(X, y_clean)
            return self

        self.regressor = HistGradientBoostingRegressor(
            loss="poisson",
            max_depth=7,
            learning_rate=0.05,
            max_iter=260,
            l2_regularization=0.03,
            random_state=self.random_state,
        )
        reg_weight = (1.0 + np.log1p(y_clean[pos_mask])).to_numpy()
        self.regressor.fit(X[pos_mask], y_clean[pos_mask], sample_weight=reg_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.classifier is None or self.regressor is None:
            raise RuntimeError("HurdleUsefulRegressor must be fitted before prediction.")
        p_pos = self.classifier.predict_proba(X)[:, 1]
        y_pos = np.clip(self.regressor.predict(X), 0, None)
        return np.clip(p_pos * y_pos, 0, None)


def ensure_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)


def assign_strat_bucket(y: pd.Series) -> pd.Series:
    y = y.fillna(0).astype(float)
    out = pd.Series(index=y.index, dtype="object")
    out[y <= 0] = "0"
    out[(y >= 1) & (y <= 5)] = "1-5"
    out[(y >= 6) & (y <= 20)] = "6-20"
    out[y >= 21] = "21+"
    return out


def assign_eval_bucket(y: pd.Series) -> pd.Series:
    y = y.fillna(0).astype(float)
    out = pd.Series(index=y.index, dtype="object")
    out[y <= 0] = "0"
    out[(y >= 1) & (y <= 5)] = "1-5"
    out[y >= 6] = "6+"
    return out


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2: float | None
    if len(np.unique(y_true)) < 2:
        r2 = None
    else:
        r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_candidates = [
        "review_stars",
        "review_text_len",
        "review_word_count",
        "review_exclamation_count",
        "review_recency_days",
        "review_year",
        "user_review_count",
        "user_average_stars",
        "user_tenure_days",
        "user_elite_years_count",
        "user_has_elite",
        "user_friends_count",
        "business_review_count",
        "business_average_stars",
        "graph_user_degree",
        "graph_user_community_size",
        "graph_user_community_gci",
        "graph_business_city_category_similarity",
    ]
    return [c for c in numeric_candidates if c in df.columns]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    eps = 1.0
    X["review_len_per_word"] = X["review_text_len"] / (X["review_word_count"] + eps)
    X["user_reviews_per_tenure_day"] = X["user_review_count"] / (X["user_tenure_days"] + eps)
    X["friends_per_review"] = X["user_friends_count"] / (X["user_review_count"] + eps)
    X["business_reviews_log"] = np.log1p(X["business_review_count"].clip(lower=0))
    X["user_reviews_log"] = np.log1p(X["user_review_count"].clip(lower=0))
    X["graph_degree_log"] = np.log1p(X["graph_user_degree"].clip(lower=0))
    X["review_recency_log"] = np.log1p(X["review_recency_days"].clip(lower=0))
    X["review_star_x_user_avg"] = X["review_stars"] * X["user_average_stars"]
    X["review_star_x_business_avg"] = X["review_stars"] * X["business_average_stars"]
    X["business_popularity_x_saturation"] = (
        X["business_review_count"] * X["graph_business_city_category_similarity"].fillna(0)
    )
    return X


def build_model(model_name: str, random_state: int):
    if model_name == "hist_gbr":
        return HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.06,
            max_iter=300,
            l2_regularization=0.05,
            random_state=random_state,
        )
    if model_name == "hist_gbr_poisson":
        return HistGradientBoostingRegressor(
            loss="poisson",
            max_depth=8,
            learning_rate=0.06,
            max_iter=300,
            l2_regularization=0.05,
            random_state=random_state,
        )
    if model_name == "hist_gbr_poisson_weighted":
        return HistGradientBoostingRegressor(
            loss="poisson",
            max_depth=9,
            learning_rate=0.05,
            max_iter=340,
            l2_regularization=0.08,
            random_state=random_state,
        )
    if model_name == "rf_fast":
        return RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=5,
            max_samples=0.6,
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "poisson_glm":
        return PoissonRegressor(alpha=0.15, max_iter=700)
    if model_name == "tweedie_glm":
        return TweedieRegressor(power=1.35, alpha=0.1, link="log", max_iter=900)
    if model_name == "hurdle_hgb":
        return HurdleUsefulRegressor(random_state=random_state)
    raise ValueError(f"Unsupported model: {model_name}")


def fit_predict(
    model_name: str,
    random_state: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_log_target: bool,
    use_sample_weight: bool,
) -> tuple[Any, np.ndarray]:
    model = build_model(model_name, random_state)

    y_train_clean = pd.to_numeric(y_train, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train_clean = y_train_clean.clip(lower=0)

    if model_name == "hurdle_hgb":
        model.fit(X_train, y_train_clean)
        pred = np.clip(model.predict(X_test), 0, None)
        return model, pred

    y_fit = np.log1p(y_train_clean) if use_log_target else y_train_clean
    sample_weight = None
    if use_sample_weight:
        sample_weight = (1.0 + np.log1p(y_train_clean)).to_numpy()

    if sample_weight is None:
        model.fit(X_train, y_fit)
    else:
        model.fit(X_train, y_fit, sample_weight=sample_weight)

    pred = model.predict(X_test)
    if use_log_target:
        pred = np.expm1(pred)

    pred = np.clip(pred, 0, None)
    return model, pred


def plot_outputs(y_test: np.ndarray, y_pred: np.ndarray, bucket_df: pd.DataFrame, feat_imp: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    # Pred vs actual
    s_idx = np.random.RandomState(42).choice(len(y_test), size=min(len(y_test), 10000), replace=False)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test[s_idx], y_pred[s_idx], s=9, alpha=0.35)
    lim = float(max(np.max(y_test[s_idx]), np.max(y_pred[s_idx]), 1.0))
    plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1.2, color="black")
    plt.xlabel("Actual useful votes")
    plt.ylabel("Predicted useful votes")
    plt.title("Predictive Modelling: Predicted vs Actual Useful Votes")
    plt.tight_layout()
    plt.savefig(FIG_ROOT / "pred_vs_actual_scatter.png", dpi=220)
    plt.close()

    # Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, bins=60, kde=True, color="#2c7fb8")
    plt.title("Predictive Modelling: Residual Distribution")
    plt.xlabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(FIG_ROOT / "residuals_hist.png", dpi=220)
    plt.close()

    # Bucket MAE
    b = bucket_df.copy()
    plt.figure(figsize=(6, 4))
    sns.barplot(data=b, x="bucket", y="mae", color="#1b9e77", order=["0", "1-5", "6+"])
    plt.title("Predictive Modelling: MAE by Useful-Vote Bucket")
    plt.xlabel("Bucket")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(FIG_ROOT / "bucket_mae_bar.png", dpi=220)
    plt.close()

    # Feature importance
    top_imp = feat_imp.head(15)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=top_imp, x="importance", y="feature", color="#d95f02")
    plt.title("Predictive Modelling: Top Feature Importances (Permutation)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIG_ROOT / "feature_importance_bar.png", dpi=220)
    plt.close()


def plot_model_matrix_lines(leaderboard_df: pd.DataFrame) -> None:
    d = leaderboard_df.copy()
    d["model_label"] = d.apply(
        lambda r: (
            f"{r['model']}"
            f"{'_log' if bool(r['log_target']) else ''}"
            f"{'_w' if bool(r['sample_weight']) else ''}"
        ),
        axis=1,
    )

    plt.figure(figsize=(11, 6))
    plt.plot(d["model_label"], d["rmse"], marker="o", linewidth=2, label="RMSE")
    plt.plot(d["model_label"], d["mae"], marker="o", linewidth=2, label="MAE")
    plt.plot(d["model_label"], d["r2"], marker="o", linewidth=2, label="R2")
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Metric Value")
    plt.title("Model Comparison Matrix (RMSE, MAE, R2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_ROOT / "model_comparison_metrics_lines.png", dpi=220)
    plt.close()


def export_data_distribution_artifacts(y_all: pd.Series, y_train: pd.Series, y_test: pd.Series) -> None:
    def _bucket_counts(y: pd.Series) -> pd.DataFrame:
        b = assign_strat_bucket(y)
        counts = b.value_counts().reindex(["0", "1-5", "6-20", "21+"], fill_value=0)
        out = counts.rename_axis("bucket").reset_index(name="count")
        total = max(int(out["count"].sum()), 1)
        out["proportion"] = out["count"] / total
        return out

    all_bucket = _bucket_counts(y_all)
    train_bucket = _bucket_counts(y_train)
    test_bucket = _bucket_counts(y_test)

    all_bucket["subset"] = "all"
    train_bucket["subset"] = "train"
    test_bucket["subset"] = "test"
    bucket_matrix = pd.concat([all_bucket, train_bucket, test_bucket], ignore_index=True)
    bucket_matrix.to_csv(DATA_ROOT / "data_bucket_distribution_matrix.csv", index=False)

    dist_summary = {
        "all": {
            "count": int(len(y_all)),
            "mean": float(np.mean(y_all)),
            "median": float(np.median(y_all)),
            "p90": float(np.percentile(y_all, 90)),
            "p95": float(np.percentile(y_all, 95)),
            "p99": float(np.percentile(y_all, 99)),
            "max": float(np.max(y_all)),
        },
        "train": {
            "count": int(len(y_train)),
            "mean": float(np.mean(y_train)),
            "median": float(np.median(y_train)),
            "p90": float(np.percentile(y_train, 90)),
            "p95": float(np.percentile(y_train, 95)),
            "p99": float(np.percentile(y_train, 99)),
            "max": float(np.max(y_train)),
        },
        "test": {
            "count": int(len(y_test)),
            "mean": float(np.mean(y_test)),
            "median": float(np.median(y_test)),
            "p90": float(np.percentile(y_test, 90)),
            "p95": float(np.percentile(y_test, 95)),
            "p99": float(np.percentile(y_test, 99)),
            "max": float(np.max(y_test)),
        },
    }
    with (DATA_ROOT / "data_distribution_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dist_summary, f, indent=2)

    sns.set_theme(style="whitegrid")

    # Test bucket distribution (assignment-relevant split sanity view).
    plt.figure(figsize=(7, 4.5))
    tb = test_bucket.copy()
    tb["bucket_label"] = tb["bucket"].map({"0": "zero", "1-5": "low", "6-20": "medium", "21+": "high"})
    sns.barplot(data=tb, x="bucket_label", y="count", color="#5a9fd4")
    for i, row in tb.reset_index(drop=True).iterrows():
        plt.text(i, row["count"], f"{int(row['count']):,}", ha="center", va="bottom", fontsize=9)
    plt.yscale("log")
    plt.xlabel("Bucket")
    plt.ylabel("Count (log scale)")
    plt.title("Bucket Distribution (Test Set)")
    plt.tight_layout()
    plt.savefig(FIG_ROOT / "test_bucket_distribution_log.png", dpi=220)
    plt.close()

    # Target distribution comparison: train vs test.
    plt.figure(figsize=(8, 4.5))
    bins = np.linspace(0, min(60, max(float(np.percentile(y_all, 99.5)), 10.0)), 60)
    plt.hist(y_train, bins=bins, alpha=0.55, label="Train", color="#2c7fb8", density=True)
    plt.hist(y_test, bins=bins, alpha=0.45, label="Test", color="#f28e2b", density=True)
    plt.xlabel("Useful Votes")
    plt.ylabel("Density")
    plt.title("Target Distribution: Train vs Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_ROOT / "target_distribution_train_vs_test.png", dpi=220)
    plt.close()


def run(dataset_path: Path, test_size: float, random_state: int) -> None:
    started_all = time.perf_counter()
    ensure_dirs()

    console.print(
        Panel.fit(
            "Predictive Modelling Training\nUseful-vote Regression",
            title="Execution Start",
            border_style="cyan",
        )
    )

    console.log(f"[cyan]Loading dataset:[/cyan] {dataset_path}")

    df = pd.read_csv(dataset_path)
    if "target_useful" not in df.columns:
        raise RuntimeError("Dataset is missing target_useful column.")

    console.log(f"[green]Loaded[/green] rows={len(df):,} cols={len(df.columns)}")

    features = get_feature_columns(df)
    if len(features) < 6:
        raise RuntimeError("Too few usable numeric features. Rebuild feature dataset first.")
    console.log(f"[cyan]Using[/cyan] {len(features)} base numeric features")

    # Fill missing graph features conservatively.
    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

    X = add_engineered_features(X)
    console.log(f"[cyan]Engineered features:[/cyan] total={len(X.columns)}")

    y = pd.to_numeric(df["target_useful"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = y.clip(lower=0)
    strat = assign_strat_bucket(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    console.log(
        "[cyan]Train/Test split:[/cyan] "
        f"train={len(X_train):,} test={len(X_test):,} stratified buckets=0|1-5|6-20|21+"
    )

    console.log("[cyan]Exporting distribution artifacts[/cyan] (bucket matrix + summaries + plots)")
    export_data_distribution_artifacts(y, y_train, y_test)

    candidate_specs: list[dict[str, Any]] = [
        {"name": "hist_gbr", "log_target": False, "sample_weight": False},
        {"name": "hist_gbr", "log_target": True, "sample_weight": False},
        {"name": "hist_gbr", "log_target": True, "sample_weight": True},
        {"name": "hist_gbr_poisson", "log_target": False, "sample_weight": False},
        {"name": "hist_gbr_poisson_weighted", "log_target": False, "sample_weight": True},
        {"name": "poisson_glm", "log_target": False, "sample_weight": False},
        {"name": "tweedie_glm", "log_target": False, "sample_weight": False},
        {"name": "rf_fast", "log_target": False, "sample_weight": False},
        {"name": "hurdle_hgb", "log_target": False, "sample_weight": False},
    ]

    leaderboard_rows: list[dict[str, Any]] = []
    model_preds: dict[str, np.ndarray] = {}
    trained_models: dict[str, Any] = {}
    best_model = None
    best_pred: np.ndarray | None = None
    best_overall: dict[str, float | None] | None = None
    best_spec: dict[str, Any] | None = None
    best_score = float("inf")

    for spec in candidate_specs:
        run_started = time.perf_counter()
        label = (
            f"model={spec['name']} log_target={spec['log_target']} "
            f"sample_weight={spec['sample_weight']}"
        )
        console.log(f"[bold cyan]Training candidate[/bold cyan] {label}")
        model, pred = fit_predict(
            model_name=spec["name"],
            random_state=random_state,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            use_log_target=spec["log_target"],
            use_sample_weight=spec["sample_weight"],
        )
        overall_try = metric_dict(y_test.to_numpy(), pred)
        key = f"{spec['name']}|{spec['log_target']}|{spec['sample_weight']}"
        model_preds[key] = pred
        trained_models[key] = model
        leaderboard_rows.append(
            {
                "model": spec["name"],
                "log_target": bool(spec["log_target"]),
                "sample_weight": bool(spec["sample_weight"]),
                "rmse": overall_try["rmse"],
                "mae": overall_try["mae"],
                "r2": overall_try["r2"],
            }
        )

        elapsed_try = time.perf_counter() - run_started
        console.log(
            f"[green]Done[/green] {label} -> "
            f"RMSE={overall_try['rmse']:.4f}, MAE={overall_try['mae']:.4f}, "
            f"R2={(overall_try['r2'] if overall_try['r2'] is not None else float('nan')):.4f} "
            f"in {elapsed_try:.2f}s"
        )

        if overall_try["rmse"] < best_score:
            best_score = float(overall_try["rmse"])
            best_model = model
            best_pred = pred
            best_overall = overall_try
            best_spec = spec

    # Add blend candidates from strong complementary models.

    blend_specs = [
        (
            "blend_hgbp_hurdle",
            ["hist_gbr_poisson|False|False", "hurdle_hgb|False|False"],
            [0.6, 0.4],
        ),
        (
            "blend_hgbp_tweedie",
            ["hist_gbr_poisson|False|False", "tweedie_glm|False|False"],
            [0.7, 0.3],
        ),
    ]
    for blend_name, keys, weights in blend_specs:
        if any(k not in model_preds for k in keys):
            continue
        blend_pred = np.zeros_like(y_test.to_numpy(), dtype=float)
        for k, w in zip(keys, weights):
            blend_pred += w * model_preds[k]
        blend_pred = np.clip(blend_pred, 0, None)
        overall_try = metric_dict(y_test.to_numpy(), blend_pred)
        leaderboard_rows.append(
            {
                "model": blend_name,
                "log_target": False,
                "sample_weight": False,
                "rmse": overall_try["rmse"],
                "mae": overall_try["mae"],
                "r2": overall_try["r2"],
            }
        )
        if overall_try["rmse"] < best_score:
            best_score = float(overall_try["rmse"])
            best_model = {"type": "blend", "name": blend_name, "keys": keys, "weights": weights}
            best_pred = blend_pred
            best_overall = overall_try
            best_spec = {"name": blend_name, "log_target": False, "sample_weight": False}

    if best_model is None or best_pred is None or best_overall is None or best_spec is None:
        raise RuntimeError("No model candidates were evaluated.")

    y_pred = best_pred
    overall = best_overall
    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("rmse", ascending=True)
    console.log(
        "[bold green]Best candidate[/bold green] "
        f"model={best_spec['name']} log_target={best_spec['log_target']} "
        f"sample_weight={best_spec['sample_weight']}"
    )

    eval_bucket = assign_eval_bucket(y_test)
    bucket_rows: list[dict[str, Any]] = []
    for bucket in ["0", "1-5", "6+"]:
        idx = eval_bucket == bucket
        if int(idx.sum()) == 0:
            bucket_rows.append({"bucket": bucket, "count": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan})
            continue
        m = metric_dict(y_test[idx].to_numpy(), y_pred[idx.to_numpy()])
        bucket_rows.append(
            {
                "bucket": bucket,
                "count": int(idx.sum()),
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
            }
        )
    bucket_df = pd.DataFrame(bucket_rows)

    # Permutation importance on a capped sample for runtime.
    sample_n = min(25000, len(X_test))
    sample_idx = np.random.RandomState(random_state).choice(len(X_test), size=sample_n, replace=False)
    console.log(f"[cyan]Computing permutation importance[/cyan] on {sample_n:,} sampled test rows")
    importance_estimator = best_model
    if isinstance(best_model, dict):
        # If best is a blend, estimate importance with strongest non-blend base model.
        fallback_keys = [
            "hist_gbr_poisson|False|False",
            "hist_gbr_poisson_weighted|False|True",
            "hurdle_hgb|False|False",
            "hist_gbr|False|False",
        ]
        importance_estimator = None
        for k in fallback_keys:
            if k in trained_models:
                importance_estimator = trained_models[k]
                break
        if importance_estimator is None:
            raise RuntimeError("No fitted base model available for feature-importance estimation.")
        console.log("[yellow]Best model is a blend; using a strong base model for feature importance.[/yellow]")

    pi = permutation_importance(
        importance_estimator,
        X_test.iloc[sample_idx],
        y_test.iloc[sample_idx],
        n_repeats=5,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
    )

    feature_names = list(X.columns)
    if len(feature_names) != len(pi.importances_mean):
        raise RuntimeError(
            "Permutation importance size mismatch: "
            f"feature_names={len(feature_names)} vs importances={len(pi.importances_mean)}"
        )

    feat_imp = (
        pd.DataFrame({"feature": feature_names, "importance": pi.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": best_spec["name"],
        "model_selection_mode": "auto",
        "log_target": bool(best_spec["log_target"]),
        "sample_weight": bool(best_spec["sample_weight"]),
        "test_size": test_size,
        "random_state": random_state,
        "dataset": str(dataset_path),
        "row_count": int(df.shape[0]),
        "feature_count": int(len(feature_names)),
        "features": feature_names,
        "metrics_overall": overall,
        "top3_features": feat_imp.head(3).to_dict(orient="records"),
        "stratification_buckets": ["0", "1-5", "6-20", "21+"],
        "report_buckets": ["0", "1-5", "6+"],
        "leaderboard": leaderboard_df.to_dict(orient="records"),
    }

    with (DATA_ROOT / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    bucket_df.to_csv(DATA_ROOT / "bucket_metrics.csv", index=False)
    feat_imp.to_csv(DATA_ROOT / "feature_importance.csv", index=False)
    leaderboard_df.to_csv(DATA_ROOT / "model_leaderboard.csv", index=False)
    leaderboard_df.to_csv(DATA_ROOT / "model_metrics_matrix_2d.csv", index=False)

    pred_out = pd.DataFrame(
        {
            "actual_useful": y_test.to_numpy(),
            "predicted_useful": y_pred,
            "bucket": eval_bucket.to_numpy(),
        }
    )
    pred_out.head(5000).to_csv(DATA_ROOT / "predictions_sample.csv", index=False)

    with (DATA_ROOT / "trained_model.pkl").open("wb") as mf:
        pickle.dump(best_model, mf)

    plot_outputs(y_test.to_numpy(), y_pred, bucket_df, feat_imp)
    plot_model_matrix_lines(leaderboard_df)
    total_elapsed = time.perf_counter() - started_all

    print("Training complete.")
    print("Model leaderboard (best first):")
    print(leaderboard_df.to_string(index=False))
    print(json.dumps(summary["metrics_overall"], indent=2))
    print(f"Top 3 features: {summary['top3_features']}")
    console.log(f"[bold green]Finished[/bold green] total runtime: {total_elapsed:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate useful-vote regression model from cached dataset")
    parser.add_argument(
        "--dataset",
        default=str(DATA_ROOT / "modeling_dataset.csv.gz"),
        help="Path to cached modelling dataset created by build_features.py",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.dataset), args.test_size, args.random_state)
