# trainer.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from tuner import tune_with_random_search, tune_with_optuna
from param_spaces import param_spaces, rf_param_space, logreg_param_space

def train_models(X_train, y_train, tune="none"):
    """
    Train models with or without hyperparameter tuning.
    tune: 'none' | 'random' | 'optuna'
    """
    results = {}

    models = {
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    print(f"\n--- Training Mode: {tune.upper()} ---")

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")

        # Basic tuning
        if tune == "random":
            best_model, best_params, best_score = tune_with_random_search(
                model,
                param_spaces.get(name, {}),
                X_train, y_train
            )

        # Advanced tuning
        elif tune == "optuna":
            if name == "RandomForestClassifier":
                best_model, best_params, best_score = tune_with_optuna(
                    RandomForestClassifier, X_train, y_train, rf_param_space
                )
            elif name == "LogisticRegression":
                best_model, best_params, best_score = tune_with_optuna(
                    LogisticRegression, X_train, y_train, logreg_param_space
                )
            else:
                model.fit(X_train, y_train)
                best_model, best_params, best_score = model, {}, model.score(X_train, y_train)

        # Normal training
        else:
            model.fit(X_train, y_train)
            best_model, best_params, best_score = model, {}, model.score(X_train, y_train)

        results[name] = {
            "model": best_model,
            "best_params": best_params,
            "score": best_score
        }

    print("\n✅ Training completed.")
    return results
