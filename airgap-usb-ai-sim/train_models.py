import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_SEED = 42


def load_datasets(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_csv = os.path.join(data_dir, "behavior_train.csv")
    test_csv = os.path.join(data_dir, "behavior_test.csv")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    return train_df, test_df


def build_feature_pipeline(numeric_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )
    return preprocessor


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    feature_cols = [
        "file_ops",
        "process_creations",
        "registry_edits",
        "network_calls",
        "avg_entropy",
        "autorun_present",
        "macro_exec",
        "signed_binary",
        "sandbox_score",
    ]
    target_col = "label"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    preprocessor = build_feature_pipeline(feature_cols)

    models = {
        "logreg": (
            Pipeline(
                steps=[
                    ("pre", preprocessor),
                    (
                        "clf",
                        LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_SEED),
                    ),
                ]
            ),
            {
                "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "clf__penalty": ["l2"],
            },
        ),
        "svm": (
            Pipeline(
                steps=[
                    ("pre", preprocessor),
                    ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
                ]
            ),
            {
                "clf__C": [0.1, 1.0, 10.0, 100.0],
                "clf__gamma": ["scale", 0.1, 0.01, 0.001],
            },
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = {}
    best_overall = {"name": None, "score": -np.inf, "model": None, "grid": None}

    for name, (pipeline, grid) in models.items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "classification_report": classification_report(y_test, y_pred, digits=4),
            "best_params": search.best_params_,
            "cv_best_score": float(search.best_score_),
        }

        results[name] = metrics

        if metrics["f1"] > best_overall["score"]:
            best_overall = {
                "name": name,
                "score": metrics["f1"],
                "model": best_model,
                "grid": search,
            }

    # Persist best model
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, f"best_model_{best_overall['name']}.joblib")
    joblib.dump(best_overall["model"], best_model_path)

    # Write report
    reports_dir = os.path.join(out_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "model_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_overall["name"],
                "best_f1": best_overall["score"],
                "saved_model_path": best_model_path,
                "all_results": results,
            },
            f,
            indent=2,
        )

    print(f"Best model: {best_overall['name']} with F1={best_overall['score']:.4f}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Detailed report: {report_path}")

    return {"results": results, "best": best_overall, "model_path": best_model_path, "report_path": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare models for malware prediction")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing behavior_train.csv and behavior_test.csv")
    parser.add_argument("--out_dir", type=str, default=".", help="Output root directory for models and reports")
    args = parser.parse_args()

    train_df, test_df = load_datasets(args.data_dir)
    train_and_evaluate(train_df, test_df, args.out_dir)


if __name__ == "__main__":
    main()


