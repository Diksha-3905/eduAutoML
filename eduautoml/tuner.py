# tuner.py
import optuna
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import numpy as np

def tune_with_random_search(model, param_grid, X_train, y_train, scoring='accuracy', cv=3, n_iter=10):
    """
    Tune model hyperparameters using RandomizedSearchCV.
    """
    if not param_grid:
        print(f"No parameter grid found for {type(model).__name__}. Using default parameters.")
        model.fit(X_train, y_train)
        return model, {}, None

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    search.fit(X_train, y_train)
    print(f"Best parameters for {type(model).__name__}: {search.best_params_}")
    print(f"Best {scoring}: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_with_optuna(model_class, X_train, y_train, param_space_func, scoring='accuracy', n_trials=30):
    """
    Advanced hyperparameter optimization using Optuna.
    """

    def objective(trial):
        params = param_space_func(trial)
        model = model_class(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best parameters for {model_class.__name__}: {study.best_params}")
    print(f"Best {scoring}: {study.best_value:.4f}")

    return model_class(**study.best_params), study.best_params, study.best_value
