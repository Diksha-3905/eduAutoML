# param_spaces.py

# For RandomizedSearchCV
param_spaces = {
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ['liblinear', 'lbfgs']
    },
    "XGBClassifier": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0]
    }
}

# For Optuna (define search functions)
def rf_param_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
    }

def logreg_param_space(trial):
    return {
        "C": trial.suggest_float("C", 0.01, 10.0, log=True),
        "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
    }
