"""
eduAutoML - Beginner-friendly, explainable AutoML for students
Main AutoML class
"""

import os
import time
import logging
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eduAutoML")


# ── Learning-mode explanations ─────────────────────────────────────────────────
LEARNING_TIPS = {
    "classification": (
        "📚 Classification predicts a CATEGORY (e.g., spam/not-spam, pass/fail). "
        "The model learns decision boundaries from labelled examples."
    ),
    "regression": (
        "📚 Regression predicts a CONTINUOUS NUMBER (e.g., house price, exam score). "
        "The model learns a mathematical function mapping inputs to outputs."
    ),
    "overfitting": (
        "⚠️  Overfitting means the model memorised the training data but can't "
        "generalise to new data. Signs: very high train accuracy but low test accuracy. "
        "Fix: use more data, regularisation, or simpler models."
    ),
    "scaling": (
        "🔧 Feature scaling (StandardScaler) makes all numeric features have mean=0 "
        "and std=1. This helps models like Logistic Regression converge faster and "
        "prevents features with large ranges from dominating."
    ),
    "encoding": (
        "🔧 Categorical columns (text labels) are converted to numbers via Label Encoding "
        "so ML algorithms can process them."
    ),
}

MODEL_REASONS = {
    "RandomForestClassifier": (
        "🌳 Random Forest was a top candidate — it handles non-linear patterns well, "
        "is robust to outliers, and rarely overfits."
    ),
    "LogisticRegression": (
        "📈 Logistic Regression performed best — your data appears linearly separable, "
        "making this fast, interpretable model ideal."
    ),
    "GradientBoostingClassifier": (
        "🚀 Gradient Boosting won — it sequentially corrects errors of weak learners, "
        "excelling on structured tabular data."
    ),
    "XGBClassifier": (
        "⚡ XGBoost took the crown — an industry-standard boosting algorithm known for "
        "high accuracy on tabular datasets."
    ),
    "LGBMClassifier": (
        "💡 LightGBM was selected — extremely fast boosting with great accuracy on "
        "medium-to-large datasets."
    ),
    "DecisionTreeClassifier": (
        "🌿 Decision Tree was chosen — simple, fully interpretable model that works well "
        "when the decision boundary is straightforward."
    ),
    "RandomForestRegressor": (
        "🌳 Random Forest Regressor led the pack — powerful ensemble that averages many "
        "trees to reduce variance in predictions."
    ),
    "Ridge": (
        "📈 Ridge Regression performed best — linear regression with L2 regularisation "
        "that prevents large coefficient values."
    ),
    "GradientBoostingRegressor": (
        "🚀 Gradient Boosting Regressor won — excellent for complex non-linear regression "
        "problems."
    ),
    "XGBRegressor": (
        "⚡ XGBoost Regressor led — high-performance boosting ideal for this regression "
        "task."
    ),
    "LGBMRegressor": (
        "💡 LightGBM Regressor was chosen — fast, accurate boosting for regression."
    ),
}


# ── Helper utilities ──────────────────────────────────────────────────────────
def _detect_task(y: pd.Series) -> str:
    if y.dtype == object or y.nunique() <= 20:
        return "classification"
    return "regression"


def _preprocess(df: pd.DataFrame, target: str, learning_mode: bool):
    if learning_mode:
        logger.info("LEARNING MODE | %s", LEARNING_TIPS["encoding"])

    df = df.copy()
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categorical target for classification
    le = None
    if y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target)

    # Encode categorical features
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_arr = imputer.fit_transform(X)

    # Scale
    if learning_mode:
        logger.info("LEARNING MODE | %s", LEARNING_TIPS["scaling"])
    scaler = StandardScaler()
    X_arr = scaler.fit_transform(X_arr)

    return pd.DataFrame(X_arr, columns=X.columns), y, le, scaler, imputer


# ── Main AutoML class ─────────────────────────────────────────────────────────
class AutoML:
    """
    eduAutoML — beginner-friendly, explainable AutoML.

    Parameters
    ----------
    learning_mode : bool
        When True, prints educational explanations at each step.
    """

    def __init__(self, learning_mode: bool = False):
        self.learning_mode = learning_mode
        self.best_model = None
        self.best_model_name = None
        self.task = None
        self.feature_names = None
        self.comparison_results = None
        self._X_test = None
        self._y_test = None
        self._le = None
        self._scaler = None
        self._imputer = None

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, target: str):
        """Preprocess, train all models, pick the best."""
        logger.info("🔍 Detecting problem type...")
        self.task = _detect_task(df[target])
        logger.info("   → Task detected: %s", self.task.upper())

        if self.learning_mode:
            logger.info("LEARNING MODE | %s", LEARNING_TIPS[self.task])

        logger.info("⚙️  Preprocessing data...")
        X, y, self._le, self._scaler, self._imputer = _preprocess(
            df, target, self.learning_mode
        )
        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if self.task == "classification" else None,
        )
        self._X_test = X_test
        self._y_test = y_test

        models = self._get_models()
        logger.info("🏋️  Training %d models...", len(models))

        results = {}
        for name, model in models.items():
            t0 = time.time()
            logger.info("   Training %s...", name)
            try:
                model.fit(X_train, y_train)
                elapsed = time.time() - t0
                score = self._score(model, X_test, y_test)
                results[name] = {"model": model, "score": score, "time": elapsed}
                logger.info("   ✅ %s → %.4f  (%.1fs)", name, score, elapsed)
            except Exception as exc:
                logger.warning("   ⚠️  %s failed: %s", name, exc)

        # Sort: classification → accuracy desc; regression → R² desc
        self.comparison_results = pd.DataFrame(
            [
                {
                    "Model": n,
                    "Score": v["score"],
                    "Time (s)": round(v["time"], 2),
                }
                for n, v in results.items()
            ]
        ).sort_values("Score", ascending=False).reset_index(drop=True)

        best_name = self.comparison_results.iloc[0]["Model"]
        self.best_model_name = best_name
        self.best_model = results[best_name]["model"]

        logger.info("\n🏆 Best model: %s (score=%.4f)", best_name, results[best_name]["score"])

        if self.learning_mode:
            reason = MODEL_REASONS.get(best_name, f"🤖 {best_name} delivered the best results.")
            logger.info("LEARNING MODE | %s", reason)
            self._check_overfitting(self.best_model, X_train, y_train, X_test, y_test)

        return self

    # ── predict ──────────────────────────────────────────────────────────────
    def predict(self, X: pd.DataFrame):
        assert self.best_model is not None, "Call fit() first."
        return self.best_model.predict(X)

    # ── explain ──────────────────────────────────────────────────────────────
    def explain(self, output_dir: str = ".", max_display: int = 10):
        """
        Generate SHAP-based explanations and feature importance charts.
        Saves plots to output_dir and returns paths.
        """
        assert self.best_model is not None, "Call fit() first."
        os.makedirs(output_dir, exist_ok=True)
        paths = []

        # Feature importance (tree-based models)
        if hasattr(self.best_model, "feature_importances_"):
            imp = pd.Series(
                self.best_model.feature_importances_, index=self.feature_names
            ).sort_values(ascending=True).tail(max_display)

            fig, ax = plt.subplots(figsize=(8, max(4, len(imp) * 0.4)))
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(imp)))
            imp.plot(kind="barh", ax=ax, color=colors)
            ax.set_title(f"Feature Importance — {self.best_model_name}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Importance Score")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            p = os.path.join(output_dir, "feature_importance.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)
            logger.info("📊 Feature importance chart saved → %s", p)

        # SHAP summary plot
        if HAS_SHAP and self._X_test is not None:
            try:
                logger.info("🔬 Computing SHAP values (this may take a moment)...")
                explainer = shap.TreeExplainer(self.best_model) if hasattr(
                    self.best_model, "feature_importances_"
                ) else shap.LinearExplainer(self.best_model, self._X_test)

                shap_values = explainer.shap_values(self._X_test)
                # For multi-class, use class 1
                sv = shap_values[1] if isinstance(shap_values, list) else shap_values

                fig, ax = plt.subplots(figsize=(9, max(4, len(self.feature_names) * 0.4)))
                shap.summary_plot(
                    sv, self._X_test,
                    feature_names=self.feature_names,
                    show=False,
                    max_display=max_display,
                    plot_size=None,
                )
                plt.title(f"SHAP Summary — {self.best_model_name}", fontsize=13, fontweight="bold")
                plt.tight_layout()
                p = os.path.join(output_dir, "shap_summary.png")
                plt.savefig(p, dpi=150, bbox_inches="tight")
                plt.close()
                paths.append(p)
                logger.info("🔬 SHAP summary saved → %s", p)
            except Exception as exc:
                logger.warning("SHAP skipped: %s", exc)
        elif not HAS_SHAP:
            logger.warning("shap not installed — run: pip install shap")

        # Confusion matrix (classification only)
        if self.task == "classification" and self._X_test is not None:
            y_pred = self.best_model.predict(self._X_test)
            cm = confusion_matrix(self._y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix — {self.best_model_name}", fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            p = os.path.join(output_dir, "confusion_matrix.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)
            logger.info("📊 Confusion matrix saved → %s", p)

        return paths

    # ── eda_report ───────────────────────────────────────────────────────────
    def eda_report(self, df: pd.DataFrame, target: str, output_dir: str = "."):
        """
        Auto EDA: missing values, distributions, correlation heatmap, class balance.
        Returns dict with stat summary and list of saved plot paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        stats = {}

        logger.info("📊 Running Auto EDA...")

        # ── 1. Overview ──────────────────────────────────────────────────────
        stats["shape"] = df.shape
        stats["missing"] = df.isnull().sum().to_dict()
        stats["dtypes"] = df.dtypes.astype(str).to_dict()

        missing_pct = df.isnull().mean() * 100
        cols_with_missing = missing_pct[missing_pct > 0]

        if not cols_with_missing.empty:
            fig, ax = plt.subplots(figsize=(8, max(3, len(cols_with_missing) * 0.5)))
            cols_with_missing.sort_values().plot(kind="barh", ax=ax, color="#e74c3c")
            ax.set_title("Missing Values (%)", fontweight="bold")
            ax.set_xlabel("% Missing")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            p = os.path.join(output_dir, "eda_missing.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)

        # ── 2. Target distribution ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        if df[target].dtype == object or df[target].nunique() <= 20:
            vc = df[target].value_counts()
            vc.plot(kind="bar", ax=ax, color=sns.color_palette("husl", len(vc)))
            ax.set_title(f"Target Distribution — '{target}'", fontweight="bold")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            plt.xticks(rotation=30, ha="right")
            stats["class_balance"] = vc.to_dict()
        else:
            ax.hist(df[target].dropna(), bins=30, color="#3498db", edgecolor="white")
            ax.set_title(f"Target Distribution — '{target}'", fontweight="bold")
            ax.set_xlabel(target)
            ax.set_ylabel("Frequency")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        p = os.path.join(output_dir, "eda_target.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # ── 3. Correlation heatmap (numeric only) ─────────────────────────────
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] > 1:
            corr = num_df.corr()
            fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.7), max(5, len(corr) * 0.6)))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr, mask=mask, annot=len(corr) <= 12, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title("Correlation Heatmap", fontweight="bold")
            plt.tight_layout()
            p = os.path.join(output_dir, "eda_correlation.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)

        # ── 4. Numeric distributions (grid) ──────────────────────────────────
        num_cols = [c for c in num_df.columns if c != target][:12]
        if num_cols:
            cols_per_row = 3
            rows = (len(num_cols) + cols_per_row - 1) // cols_per_row
            fig, axes = plt.subplots(rows, cols_per_row, figsize=(14, rows * 3))
            axes = np.array(axes).flatten()
            for i, col in enumerate(num_cols):
                axes[i].hist(df[col].dropna(), bins=25, color="#2ecc71", edgecolor="white", alpha=0.85)
                axes[i].set_title(col, fontsize=9)
                axes[i].spines[["top", "right"]].set_visible(False)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.suptitle("Numeric Feature Distributions", fontweight="bold", y=1.01)
            plt.tight_layout()
            p = os.path.join(output_dir, "eda_distributions.png")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)

        logger.info("✅ EDA complete — %d charts saved to %s", len(paths), output_dir)
        return stats, paths

    # ── compare_models ───────────────────────────────────────────────────────
    def compare_models(self, output_dir: str = "."):
        """Return comparison DataFrame and save bar chart."""
        assert self.comparison_results is not None, "Call fit() first."
        os.makedirs(output_dir, exist_ok=True)

        df = self.comparison_results.copy()
        metric_label = "Accuracy" if self.task == "classification" else "R² Score"
        df = df.rename(columns={"Score": metric_label})

        fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.6)))
        colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(df))]
        bars = ax.barh(df["Model"][::-1], df[metric_label][::-1], color=colors[::-1])
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
        ax.set_title(f"Model Comparison — {metric_label}", fontweight="bold", fontsize=13)
        ax.set_xlabel(metric_label)
        ax.set_xlim(0, df[metric_label].max() * 1.15)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        p = os.path.join(output_dir, "model_comparison.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)

        logger.info("📊 Model comparison chart saved → %s", p)
        return df, p

    # ── save / load ──────────────────────────────────────────────────────────
    def save(self, path: str = "automl_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("💾 Model saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "AutoML":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("📦 Model loaded from %s", path)
        return obj

    # ── private helpers ──────────────────────────────────────────────────────
    def _get_models(self):
        if self.task == "classification":
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
                "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            }
            if HAS_XGB:
                models["XGBClassifier"] = XGBClassifier(
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=42, verbosity=0,
                )
            if HAS_LGBM:
                models["LGBMClassifier"] = LGBMClassifier(random_state=42, verbose=-1)
        else:
            models = {
                "Ridge": Ridge(),
                "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
            }
            if HAS_XGB:
                models["XGBRegressor"] = XGBRegressor(random_state=42, verbosity=0)
            if HAS_LGBM:
                models["LGBMRegressor"] = LGBMRegressor(random_state=42, verbose=-1)
        return models

    def _score(self, model, X_test, y_test):
        if self.task == "classification":
            return accuracy_score(y_test, model.predict(X_test))
        else:
            return r2_score(y_test, model.predict(X_test))

    def _check_overfitting(self, model, X_train, y_train, X_test, y_test):
        train_s = self._score(model, X_train, y_train)
        test_s = self._score(model, X_test, y_test)
        gap = train_s - test_s
        if gap > 0.15:
            logger.info(
                "LEARNING MODE | %s\n   Train=%.3f  Test=%.3f  Gap=%.3f",
                LEARNING_TIPS["overfitting"], train_s, test_s, gap,
            )
        else:
            logger.info(
                "LEARNING MODE | ✅ No signs of overfitting! Train=%.3f  Test=%.3f",
                train_s, test_s,
            )
